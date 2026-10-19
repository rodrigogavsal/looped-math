"""Calibración del dataset con un GPT normal (Parte 1).

Entrena un GPT pequeño (por defecto 4 capas, emb_dim 256, 8 cabezas) sobre
data/train.jsonl y mide la exactitud por nivel en data/val.jsonl. Sirve para
decidir si la dificultad del dataset es la adecuada ANTES de lanzar la matriz
de experimentos. Objetivo aproximado: ~95 % en el nivel 1 y 20-40 % en el nivel 4.

Uso (desde la raíz del repositorio):
    python -m src.calibrate                              # GPT de 4 capas, 3000 pasos
    python -m src.calibrate --steps 6000                 # más largo, si la curva aún sube
    python -m src.calibrate --layers 1 --emb_dim 32 --steps 30 --batch 8 --eval_limit 20   # prueba de humo (CPU)

El modelo es el GPTModel del notebook (Raschka, cap. 4) copiado aquí para que la
calibración no dependa de la Parte 2. Cuando exista src/model.py, se puede usar ese.
La atención y la GELU usan las versiones optimizadas de PyTorch (mismo cálculo, mucha
menos memoria): con la atención explícita del notebook, batch 64 no cabe en 4 GB.

Si aun así no cabe en memoria: --batch 32 --steps 6000 (mismo número de ejemplos vistos).

Exactitud: se mide con teacher forcing, exigiendo que el argmax sea correcto en
TODOS los caracteres de la respuesta y en el <eos>. Es exactamente equivalente a
generar de forma greedy y comparar el número (si el modelo acierta cada carácter
dado el prefijo correcto, el greedy produce justo ese prefijo), pero mucho más rápido.
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import CharTokenizer, MAX_LEN, format_example, format_prompt

IGNORE = -100


# ============================================================================ modelo (notebook, sección 3-5)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out, self.num_heads, self.head_dim = d_out, num_heads, d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, T, _ = x.shape
        q = self.W_query(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Misma operación que el notebook (softmax(QK^T/sqrt(d) + máscara causal) V), pero con el
        # kernel optimizado de PyTorch: no guarda las matrices T x T y gasta mucha menos memoria.
        # (Para analizar atenciones hace falta la versión explícita; aquí solo se entrena.)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(b, T, self.d_out)
        return self.out_proj(out)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    """Misma aproximación con tanh que el notebook, en la versión fusionada de PyTorch (menos memoria)."""

    def forward(self, x):
        return F.gelu(x, approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.layers = nn.Sequential(nn.Linear(d, 4 * d), GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"],
                                      cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1, self.norm2 = LayerNorm(cfg["emb_dim"]), LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        return x + self.drop_shortcut(self.ff(self.norm2(x)))


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.trf_blocks(self.drop_emb(x))
        return self.out_head(self.final_norm(x))


# ============================================================================ datos
class EncodedSplit:
    """Un fichero JSONL tokenizado y guardado de forma compacta (un tensor plano + offsets)."""

    def __init__(self, path, tok, limit_per_level=None, levels=None, reverse=False):
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        if levels:
            rows = [r for r in rows if r["level"] in levels]
        if limit_per_level:
            kept, count = [], defaultdict(int)
            for r in rows:
                if count[r["level"]] < limit_per_level:
                    kept.append(r)
                    count[r["level"]] += 1
            rows = kept
        self.rows = rows
        seqs, self.prompt_len = [], []
        for r in rows:
            answer = str(r["answer"])[::-1] if reverse else r["answer"]  # cifras al revés (opcional)
            ids = tok.encode(format_example(r["question"], answer), add_eos=True)
            assert len(ids) <= MAX_LEN
            seqs.append(ids)
            self.prompt_len.append(len(format_prompt(r["question"])))
        self.lengths = torch.tensor([len(s) for s in seqs])
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), self.lengths.cumsum(0)])
        self.flat = torch.tensor([i for s in seqs for i in s], dtype=torch.uint8)
        self.prompt_len = torch.tensor(self.prompt_len)

    def __len__(self):
        return len(self.rows)

    def batch(self, idx, pad_id, device, with_question=False):
        """x, y para los índices dados. y = -100 en la pregunta y en el relleno (pérdida solo en la respuesta).

        Con with_question=True devuelve además yq: los objetivos de la PREGUNTA (siguiente carácter
        del enunciado), con -100 en la respuesta y en el relleno. Sirve para la pérdida auxiliar.
        """
        L = int(self.lengths[idx].max()) - 1
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        y = torch.full((len(idx), L), IGNORE, dtype=torch.long)
        yq = torch.full((len(idx), L), IGNORE, dtype=torch.long) if with_question else None
        for j, i in enumerate(idx):
            s = self.flat[self.offsets[i]:self.offsets[i + 1]].long()
            n, p = len(s) - 1, int(self.prompt_len[i]) - 1
            x[j, :n] = s[:-1]
            y[j, :n] = s[1:]
            if with_question:
                yq[j, :p] = s[1:p + 1]
            y[j, :p] = IGNORE  # solo se predicen la respuesta y el <eos>
        if with_question:
            return x.to(device), y.to(device), yq.to(device)
        return x.to(device), y.to(device)


# ============================================================================ evaluación
@torch.no_grad()
def evaluate(model, split, pad_id, device, ctx, batch_size=128):
    """Exactitud exacta y entropía cruzada de la respuesta, por nivel y por tipo."""
    model.eval()
    correct, ce_sum, tok_n = defaultdict(list), defaultdict(float), defaultdict(int)
    order = sorted(range(len(split)), key=lambda i: int(split.lengths[i]))  # agrupar por longitud: menos relleno
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        x, y = split.batch(idx, pad_id, device)
        with ctx:
            logits = model(x)
        logits = logits.float()
        valid = y != IGNORE
        ok = ((logits.argmax(-1) == y) | ~valid).all(dim=1)
        ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE, reduction="none").view_as(y)
        for j, i in enumerate(idx):
            r = split.rows[i]
            group = "A" if r["type"] == "A" else ("B desordenado" if r["shuffled"] else "B ordenado")
            for key in (("level", r["level"]), ("type", group), ("all", "all")):
                correct[key].append(bool(ok[j]))
                ce_sum[key] += float(ce[j][valid[j]].sum())
                tok_n[key] += int(valid[j].sum())
    model.train()
    return {f"{k[0]}={k[1]}": {"acc": sum(v) / len(v), "ce": ce_sum[k] / tok_n[k], "n": len(v)}
            for k, v in sorted(correct.items(), key=lambda kv: str(kv[0]))}


@torch.no_grad()
def greedy_answer(model, tok, question, device, max_new=5):
    model.eval()
    ids = torch.tensor([tok.encode(format_prompt(question))], device=device)
    out = []
    for _ in range(max_new):
        nxt = int(model(ids)[0, -1].argmax())
        if nxt == tok.eos_id:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    model.train()
    return tok.decode(out)


# ============================================================================ entrenamiento
def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def print_table(res, title):
    levels = [k for k in res if k.startswith("level=")]
    types = [k for k in res if k.startswith("type=")]
    print(f"  {title}")
    print("   " + "  ".join(f"{k.split('=')[1]:>6s}" for k in ["level=" + k.split("=")[1] for k in levels])
          + f"  {'total':>6s}   |  " + "  ".join(f"{k.split('=')[1]}" for k in types))
    print("   " + "  ".join(f"{100 * res[k]['acc']:5.1f}%" for k in levels) + f"  {100 * res['all=all']['acc']:5.1f}%"
          + "   |  " + "  ".join(f"{100 * res[k]['acc']:.1f}%" for k in types))


def main():
    ap = argparse.ArgumentParser(description="Calibra la dificultad del dataset con un GPT normal.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_limit", type=int, default=None, help="problemas de val por nivel (por defecto, todos)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", type=int, nargs="+", default=None, help="entrenar y evaluar solo estos niveles")
    ap.add_argument("--q_weight", type=float, default=0.0,
                    help="peso de la pérdida auxiliar sobre el texto de la pregunta (0 = solo la respuesta)")
    ap.add_argument("--threads", type=int, default=None, help="hilos de CPU (solo en CPU)")
    ap.add_argument("--reverse_answer", action="store_true",
                    help="el modelo escribe la respuesta con las cifras al revés (unidades primero)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out", default="results/calibration")
    ap.add_argument("--load", default=None,
                    help="checkpoint .pt ya entrenado: no entrena, solo evalúa en val y en los test")
    ap.add_argument("--no_tests", action="store_true", help="no evaluar test_id / test_tpl / test_ood al final")
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = torch.device(a.device)
    tok = CharTokenizer()
    print(f"Cargando datos de {a.data}/ ...")
    t0 = time.time()
    if a.threads:
        torch.set_num_threads(a.threads)
    train = EncodedSplit(os.path.join(a.data, "train.jsonl"), tok, levels=a.levels, reverse=a.reverse_answer)
    val = EncodedSplit(os.path.join(a.data, "val.jsonl"), tok, a.eval_limit, levels=a.levels, reverse=a.reverse_answer)
    majority = max(set(r["answer"] for r in train.rows), key=[r["answer"] for r in train.rows].count)
    baseline = sum(r["answer"] == majority for r in val.rows) / len(val)
    print(f"  referencia: contestar siempre {majority} acierta el {100 * baseline:.1f} % de val")
    print(f"  train={len(train)}  val={len(val)}  ({time.time() - t0:.0f}s)")

    cfg = {"vocab_size": tok.vocab_size, "context_length": MAX_LEN, "emb_dim": a.emb_dim, "n_heads": a.heads,
           "n_layers": a.layers, "drop_rate": 0.0, "qkv_bias": False}
    model = GPTModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT de {a.layers} capas, emb_dim={a.emb_dim}, {a.heads} cabezas: {n_params:,} parámetros, dispositivo {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    use_amp = device.type == "cuda" and not a.no_amp
    bf16 = use_amp and torch.cuda.is_bf16_supported()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda") if use_amp and not bf16 else None
    print(f"  precisión: {'bf16' if bf16 else ('fp16 + GradScaler' if use_amp else 'fp32')}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    curve = []
    t0 = time.time()
    model.train()
    if a.load:
        model.load_state_dict(torch.load(a.load, map_location=device))
        print(f"  checkpoint cargado: {a.load} (no se entrena)")
    for step in range(0 if a.load else a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
        idx = random.sample(range(len(train)), a.batch)
        if a.q_weight > 0:
            x, y, yq = train.batch(idx, tok.pad_id, device, with_question=True)
        else:
            x, y = train.batch(idx, tok.pad_id, device)
        with ctx:
            logits = model(x)
        logits = logits.float().flatten(0, 1)
        loss = F.cross_entropy(logits, y.flatten(), ignore_index=IGNORE)
        answer_loss = loss
        if a.q_weight > 0:
            loss = loss + a.q_weight * F.cross_entropy(logits, yq.flatten(), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            res = evaluate(model, val, tok.pad_id, device, ctx)
            elapsed = time.time() - t0
            curve.append({"step": step + 1, "train_loss": loss.item(), "answer_loss": answer_loss.item(),
                          "time_s": elapsed,
                          **{k: v["acc"] for k, v in res.items()}})
            print(f"\npaso {step + 1}/{a.steps}  pérdida de la respuesta {answer_loss.item():.3f}  ({elapsed / 60:.1f} min)")
            print_table(res, "exactitud en val por nivel  |  por tipo")

    if a.load:
        res = evaluate(model, val, tok.pad_id, device, ctx)
        print_table(res, "exactitud en val por nivel  |  por tipo")

    tests = {}
    if not a.no_tests:
        for name, desc in (("test_id", "mismos tipos y frases que train"),
                           ("test_tpl", "frases NUEVAS (redacciones reservadas)"),
                           ("test_ood", "niveles 5-6, más largos que train")):
            path = os.path.join(a.data, f"{name}.jsonl")
            if not os.path.exists(path):
                continue
            split = EncodedSplit(path, tok, levels=a.levels, reverse=a.reverse_answer)
            if len(split) == 0:
                continue
            tests[name] = evaluate(model, split, tok.pad_id, device, ctx, batch_size=64)
            print(f"\n{name}: {desc}")
            print_table(tests[name], f"exactitud en {name} por nivel  |  por tipo")

    print("\nEjemplos (generación greedy):")
    for r in random.sample(val.rows, 5):
        pred = greedy_answer(model, tok, r["question"], device)
        if a.reverse_answer:
            pred = pred[::-1]
        mark = "OK " if pred == str(r["answer"]) else "MAL"
        print(f"  [{mark}] nivel {r['level']} tipo {r['type']}: modelo={pred!r} correcta={r['answer']}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tag = ((f"_lv{''.join(map(str, a.levels))}" if a.levels else "") + (f"_q{a.q_weight:g}" if a.q_weight else "")
           + ("_rev" if a.reverse_answer else ""))
    out_path = f"{a.out}_L{a.layers}_d{a.emb_dim}{tag}_s{a.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "params": n_params, "baseline_majority": baseline, "final": res, "curve": curve,
                   "tests": tests, "train_time_s": time.time() - t0}, f, indent=1, ensure_ascii=False)
    print(f"\nResultados guardados en {out_path}")
    if not a.load:
        ckpt_path = out_path[:-len(".json")] + ".pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint guardado en {ckpt_path}  (reevaluar sin entrenar: --load {ckpt_path})")
    acc = {int(k.split("=")[1]): v["acc"] for k, v in res.items() if k.startswith("level=")}
    print(f"\nLectura rápida (objetivo: nivel 1 ~95 %, nivel 4 entre 20 y 40 %; azar = {100 * baseline:.1f} %):")
    top = max(acc)
    if acc.get(1, 0) < 0.85:
        print("  El nivel 1 sale bajo: o el dataset es demasiado difícil o falta entrenamiento (mira si la curva aún sube).")
    elif acc[top] > 0.6:
        print(f"  El nivel {top} sale muy alto: el dataset es demasiado fácil para un GPT de {a.layers} capas.")
    elif acc[top] < 1.5 * baseline:
        print(f"  El nivel {top} está cerca del azar: el dataset es demasiado difícil para este modelo y presupuesto.")
    elif 0.2 <= acc[top] <= 0.4:
        print("  Reparto razonable: hay margen para que los modelos más profundos se distingan.")
    else:
        print(f"  El nivel {top} queda fuera del rango 20-40 %, pero por encima del azar: revisa las curvas.")


if __name__ == "__main__":
    main()
