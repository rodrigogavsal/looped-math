"""Desglose de la exactitud por redacción (plantilla) con un checkpoint ya entrenado (Parte 1).

Responde a: "en test_tpl, ¿qué redacciones reservadas generaliza el modelo y cuáles no?".
No entrena nada: carga el .pt que guarda `src.calibrate` y evalúa en unos segundos.

Uso (desde la raíz del repositorio):
    python -m src.eval_templates --data data --load results/full_L4_d256_s0.pt
    # si el modelo no es el de por defecto (4 capas, 256, 8 cabezas): --layers --emb_dim --heads

Qué imprime:
  1. Exactitud por nivel en test_id y test_tpl (debe coincidir con la de calibrate.py).
  2. Para cada una de las 18 redacciones reservadas (3 marcos x 6 operaciones):
       - exactitud en test_tpl de los problemas que la usan,
       - exactitud en test_tpl cuando es la ÚNICA redacción reservada del problema,
       - exactitud solo en nivel 1 (una sola operación: la medida más limpia),
       - referencia: exactitud en nivel 1 de test_id en problemas con la MISMA operación y marco
         pero redacciones vistas en train.
     La diferencia entre las dos últimas columnas es lo que cuesta la redacción nueva.
  3. Las redacciones de train que peor sale en test_id (para ver si algo ya es difícil de por sí).
Guarda todo en <out>_templates.json.
"""
import argparse
import json
import os
from collections import defaultdict
from contextlib import nullcontext

import torch

from . import generate_data as gd
from .calibrate import EncodedSplit, GPTModel, IGNORE
from .tokenizer import CharTokenizer, MAX_LEN

FRAMES = {"A_PERSON": "persona", "A_CONTAINER": "recipiente", "B_CHAIN": "tipo B"}
OPS = {"add": "sumar", "sub": "restar", "mul2": "doble", "mul3": "triple", "half": "mitad", "div": "parte"}


def template_index():
    """id de redacción de operación -> (marco, operación, estilo, texto)."""
    index = {}
    for frame, table in (("A_PERSON", gd.A_PERSON), ("A_CONTAINER", gd.A_CONTAINER), ("B_CHAIN", gd.B_CHAIN)):
        for op, lst in table["ops"].items():
            for tid, style, _held, text, _flags in lst:
                index[tid] = (FRAMES[frame], OPS[op], "compacta" if style == "c" else "larga", text)
    return index


@torch.no_grad()
def correct_per_row(model, split, pad_id, device, ctx, batch_size=64):
    """Lista de bool (uno por problema): ¿acierta todos los caracteres de la respuesta y el <eos>?"""
    model.eval()
    out = [False] * len(split)
    order = sorted(range(len(split)), key=lambda i: int(split.lengths[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        x, y = split.batch(idx, pad_id, device)
        with ctx:
            logits = model(x)
        valid = y != IGNORE
        ok = ((logits.float().argmax(-1) == y) | ~valid).all(dim=1)
        for j, i in enumerate(idx):
            out[i] = bool(ok[j])
    return out


def acc(flags):
    return (sum(flags) / len(flags)) if flags else float("nan")


def fmt(a, n):
    return f"{100 * a:5.1f} % (n={n})" if n else "   —"


def main():
    ap = argparse.ArgumentParser(description="Exactitud por redacción reservada con un checkpoint entrenado.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--load", required=True, help="checkpoint .pt guardado por src.calibrate")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out", default=None, help="prefijo de salida (por defecto, el del checkpoint)")
    a = ap.parse_args()

    device = torch.device(a.device)
    tok = CharTokenizer()
    cfg = {"vocab_size": tok.vocab_size, "context_length": MAX_LEN, "emb_dim": a.emb_dim, "n_heads": a.heads,
           "n_layers": a.layers, "drop_rate": 0.0, "qkv_bias": False}
    model = GPTModel(cfg).to(device)
    model.load_state_dict(torch.load(a.load, map_location=device))
    use_amp = device.type == "cuda" and not a.no_amp
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
           if use_amp else nullcontext())
    print(f"Checkpoint cargado: {a.load}  ({sum(p.numel() for p in model.parameters()):,} parámetros)")

    index = template_index()
    held = set(gd.HELD_IDS)
    results = {}
    flags = {}
    for name in ("test_id", "test_tpl"):
        split = EncodedSplit(os.path.join(a.data, f"{name}.jsonl"), tok)
        ok = correct_per_row(model, split, tok.pad_id, device, ctx)
        flags[name] = [(r, o) for r, o in zip(split.rows, ok)]

    # ---- 1. sanity: exactitud por nivel
    print("\n1) Exactitud por nivel")
    results["por_nivel"] = {}
    for name, rows in flags.items():
        by = defaultdict(list)
        for r, o in rows:
            by[r["level"]].append(o)
        results["por_nivel"][name] = {lv: acc(v) for lv, v in sorted(by.items())}
        print(f"   {name:9s}" + "  ".join(f"nivel {lv}: {100 * acc(v):5.1f} %" for lv, v in sorted(by.items()))
              + f"   total: {100 * acc([o for _, o in rows]):5.1f} %")

    # ---- 2. por redacción reservada
    def op_of(row):
        """(marco, operación) de los problemas de nivel 1 (tienen una sola operación)."""
        ids = [t for t in row["template"].split("|")[1:] if t in index]
        return (index[ids[0]][0], index[ids[0]][1]) if len(ids) == 1 else None

    ref = defaultdict(list)  # (marco, operación) -> aciertos en nivel 1 de test_id
    for r, o in flags["test_id"]:
        if r["level"] == 1 and op_of(r):
            ref[op_of(r)].append(o)

    print("\n2) Redacciones reservadas (solo aparecen en test_tpl)")
    print(f"   {'redacción':10s} {'marco':10s} {'operación':9s} {'todos los niveles':22s} {'única reservada':22s} "
          f"{'nivel 1 (reservada)':22s} {'nivel 1 (ref. vistas)':22s} texto")
    table = []
    for tid in sorted(held, key=lambda t: (index[t][0], index[t][1])):
        frame, op, style, text = index[tid]
        uses = [(r, o) for r, o in flags["test_tpl"] if tid in r["template"].split("|")[1:]]
        only = [(r, o) for r, o in uses if sum(t in held for t in r["template"].split("|")[1:]) == 1]
        l1 = [o for r, o in uses if r["level"] == 1]
        rf = ref[(frame, op)]
        row = {"id": tid, "marco": frame, "operacion": op, "estilo": style, "texto": text,
               "acc_todos": acc([o for _, o in uses]), "n_todos": len(uses),
               "acc_unica": acc([o for _, o in only]), "n_unica": len(only),
               "acc_nivel1": acc(l1), "n_nivel1": len(l1),
               "ref_nivel1": acc(rf), "n_ref_nivel1": len(rf)}
        table.append(row)
        print(f"   {tid:10s} {frame:10s} {op:9s} {fmt(row['acc_todos'], row['n_todos']):22s} "
              f"{fmt(row['acc_unica'], row['n_unica']):22s} {fmt(row['acc_nivel1'], row['n_nivel1']):22s} "
              f"{fmt(row['ref_nivel1'], row['n_ref_nivel1']):22s} {text}")
    results["reservadas"] = table
    print("   Ojo con n: con n < 50 la exactitud tiene un error de ±10 puntos o más; compara solo diferencias grandes.")

    # ---- 3. redacciones de train más difíciles en test_id
    per_tid = defaultdict(list)
    for r, o in flags["test_id"]:
        for t in r["template"].split("|")[1:]:
            if t in index:
                per_tid[t].append(o)
    hardest = sorted(((acc(v), len(v), t) for t, v in per_tid.items() if len(v) >= 100))[:8]
    print("\n3) Redacciones vistas en train con peor exactitud en test_id (n >= 100; mezcla niveles)")
    for a_, n, t in hardest:
        frame, op, style, text = index[t]
        print(f"   {t:12s} {frame:10s} {op:9s} {100 * a_:5.1f} % (n={n})  {text}")
    results["peores_vistas"] = [{"id": t, "acc": a_, "n": n} for a_, n, t in hardest]

    out = (a.out or os.path.splitext(a.load)[0]) + "_templates.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
