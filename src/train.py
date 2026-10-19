"""Bucle de entrenamiento y gestión de checkpoints según la Sección 2.5, 2.7 y Parte 3 del contrato común.

Maneja:
- Hiperparámetros fijados en base.yaml (AdamW, lr 1e-3, warmup 200 pasos + coseno al 10%)
- Recorte de gradientes a 1.0
- Ahorro de memoria opcional, sin cambiar el resultado:
    grad_accum: N  -> cada paso se hace en N trozos de batch_size/N ejemplos (mismo batch efectivo)
    grad_checkpoint: true -> recalcula las activaciones de cada bloque en el backward
- Precisión mixta automática (bf16 si soporta, fp16 con GradScaler si no)
- Semillas fijadas (torch, numpy, random)
- Evaluación rápida en validación cada 250 pasos (entropía cruzada y teacher forcing)
- Guardado de curva y checkpoints: checkpoints/<run>.pt
- Evaluación completa al finalizar y guardado en results/<run>.json con el esquema exacto de la Sección 2.7
- Prueba de cordura: sobreajuste en 32 ejemplos hasta pérdida ~0 (< 0.05)
"""

import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import IGNORE_INDEX, MathDataset, create_dataloader
from src.evaluate import print_eval_samples, run_full_evaluation
from src.model import build_model, count_params, flops_per_token
from src.tokenizer import CharTokenizer, get_tokenizer


def set_seed(seed: int):
    """Fija las semillas para reproducibilidad estricta."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Warmup lineal durante `warmup_steps` y decaimiento coseno hasta el 10 % del lr base."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def get_device() -> torch.device:
    """Detecta el mejor dispositivo disponible: CUDA, MPS (Apple Silicon), o CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_context(device: torch.device, precision: str = "auto") -> Tuple[Any, Optional[torch.amp.GradScaler]]:
    """Configura precisión mixta según el dispositivo y las especificaciones del contrato."""
    if device.type == "cuda":
        if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16), None
        elif precision in ("fp16", "auto"):
            scaler = torch.amp.GradScaler("cuda")
            return torch.autocast(device_type="cuda", dtype=torch.float16), scaler
    # En CPU o MPS usamos fp32 sin scaler
    return torch.autocast(device_type="cpu", enabled=False), None


@torch.no_grad()
def evaluate_val_quick(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 8,
) -> Tuple[float, float]:
    """Evaluación rápida de validación para seguimiento del entrenamiento."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    batch_count = 0
    for batch in val_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        logits = model(x)
        # Pérdida sobre los tokens que no son IGNORE_INDEX
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=IGNORE_INDEX, reduction="sum")
        mask = y != IGNORE_INDEX
        n_tokens = mask.sum().item()

        if n_tokens > 0:
            total_loss += loss.item()
            total_tokens += n_tokens

            preds = logits.argmax(dim=-1)
            correct_tokens += (preds[mask] == y[mask]).sum().item()

        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    val_ce = total_loss / max(1, total_tokens)
    val_acc = correct_tokens / max(1, total_tokens)
    return round(val_ce, 4), round(val_acc, 4)


def train_run(
    cfg: Dict[str, Any],
    train_source: Union[str, List[Dict[str, Any]]],
    val_source: Union[str, List[Dict[str, Any]]],
    test_id_source: Union[str, List[Dict[str, Any]]],
    test_ood_source: Union[str, List[Dict[str, Any]]],
    tokenizer: Optional[CharTokenizer] = None,
    device: Optional[torch.device] = None,
    run_name: Optional[str] = None,
    checkpoints_dir: str = "checkpoints",
    results_dir: str = "results",
    verbose: bool = True,
    test_tpl_source: Optional[Union[str, List[Dict[str, Any]]]] = None,
) -> Tuple[Dict[str, Any], nn.Module]:
    """Entrena un run de la matriz de principio a fin, lo evalúa y guarda los resultados."""
    if device is None:
        device = get_device()
    if tokenizer is None:
        tokenizer = get_tokenizer()

    seed = cfg.get("seed", 0)
    set_seed(seed)

    # Actualizar vocab_size en la configuración
    cfg["vocab_size"] = tokenizer.vocab_size

    if run_name is None:
        arch = cfg.get("arch", "looped")
        depth = cfg.get("n_layers") if arch == "gpt" else cfg.get("n_loops", 4)
        prefix = "gpt_L" if arch == "gpt" else "looped_N"
        d = cfg.get("emb_dim", 256)
        run_name = f"{prefix}{depth}_d{d}_s{seed}"

    # Construir modelo
    model = build_model(cfg).to(device)
    model.grad_checkpoint = bool(cfg.get("grad_checkpoint", False))
    n_params = count_params(model)
    flops = flops_per_token(cfg)

    if verbose:
        print(f"\n[{run_name}] Modelo {cfg.get('arch').upper()} construido:")
        print(f"  Parámetros: {n_params:,}")
        print(f"  FLOPs/token aprox.: {flops:,.0f}")
        print(f"  Dispositivo: {device}")

    # DataLoaders
    batch_size = cfg.get("batch_size", 64)
    grad_accum = int(cfg.get("grad_accum", 1))
    if grad_accum < 1 or batch_size % grad_accum != 0:
        raise ValueError(f"batch_size={batch_size} tiene que ser múltiplo de grad_accum={grad_accum}")
    micro_batch = batch_size // grad_accum
    if verbose and (grad_accum > 1 or model.grad_checkpoint):
        print(f"  Ahorro de memoria: {grad_accum} trozos de {micro_batch} ejemplos por paso"
              f"{', checkpointing de activaciones' if model.grad_checkpoint else ''}")
    train_loader = create_dataloader(
        train_source, tokenizer=tokenizer, batch_size=micro_batch, shuffle=True
    )
    val_loader = create_dataloader(
        val_source, tokenizer=tokenizer, batch_size=batch_size, shuffle=False
    )

    # Optimizador según Sección 2.5: AdamW, lr 1e-3, betas (0.9, 0.95), weight_decay 0.1
    base_lr = cfg.get("lr", 1e-3)
    betas = tuple(cfg.get("betas", [0.9, 0.95]))
    weight_decay = cfg.get("weight_decay", 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=betas, weight_decay=weight_decay)

    warmup_steps = cfg.get("warmup_steps", 200)
    total_steps = cfg.get("total_steps", cfg.get("steps", 10000))
    grad_clip = cfg.get("grad_clip", 1.0)
    eval_every = cfg.get("eval_every", 250)
    precision = cfg.get("precision", "auto")

    autocast_ctx, scaler = get_autocast_context(device, precision=precision)

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    curve: List[Dict[str, Any]] = []
    t0 = time.time()
    train_iter = iter(train_loader)

    model.train()
    if verbose:
        print(f"  Iniciando entrenamiento ({total_steps} pasos)...")

    for step in range(total_steps):
        # Actualizar LR con schedule
        current_lr = lr_at(step, total_steps, base_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Obtener los trozos del batch (bucle infinito sobre el dataset de entrenamiento).
        # Con grad_accum = 1 es un único batch de batch_size ejemplos, como antes.
        micro_batches = []
        for _ in range(grad_accum):
            try:
                micro_batches.append(next(train_iter))
            except StopIteration:
                train_iter = iter(train_loader)
                micro_batches.append(next(train_iter))

        # La pérdida es la media sobre TODOS los tokens de respuesta del batch completo:
        # cada trozo suma su pérdida y se divide por el total de tokens válidos del batch.
        # Así el gradiente es el mismo que con un único batch de batch_size ejemplos.
        n_valid = sum(int((mb["y"] != IGNORE_INDEX).sum()) for mb in micro_batches)

        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for mb in micro_batches:
            x = mb["x"].to(device)
            y = mb["y"].to(device)
            with autocast_ctx:
                logits = model(x)
                loss_sum = F.cross_entropy(
                    logits.view(-1, cfg["vocab_size"]).float(), y.view(-1),
                    ignore_index=IGNORE_INDEX, reduction="sum",
                )
                loss = loss_sum / max(1, n_valid)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_total += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        loss = torch.tensor(loss_total)

        # Evaluación periódica en validación
        if step % eval_every == 0 or step == total_steps - 1:
            val_ce, val_acc = evaluate_val_quick(model, val_loader, device)
            model.train()
            elapsed = time.time() - t0
            record_point = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_ce_answer": val_ce,
                "val_acc": val_acc,
            }
            curve.append(record_point)
            if verbose:
                print(
                    f"  Paso {step:5d}/{total_steps} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Val CE: {val_ce:.4f} | "
                    f"Val Acc (TF): {val_acc*100:.1f}% | "
                    f"LR: {current_lr:.2e} | "
                    f"Tiempo: {elapsed:.1f}s"
                )

    train_time_s = round(time.time() - t0, 2)

    # Guardar checkpoint definitivo según Sección 2.7
    ckpt_path = os.path.join(checkpoints_dir, f"{run_name}.pt")
    torch.save(
        {
            "cfg": cfg,
            "model_state": model.state_dict(),
            "step": total_steps,
        },
        ckpt_path,
    )
    if verbose:
        print(f"  Checkpoint guardado en: {ckpt_path}")

    # Evaluación completa al finalizar (generación greedy + loops sweep)
    if verbose:
        print("  Ejecutando evaluación completa (test_id, test_tpl, test_ood, loops_sweep)...")

    eval_results = run_full_evaluation(
        model=model,
        test_id_source=test_id_source,
        test_ood_source=test_ood_source,
        tokenizer=tokenizer,
        device=device,
        run_loops_sweep=(cfg.get("arch", "looped") == "looped"),
        test_tpl_source=test_tpl_source,
        preds_path=os.path.join(results_dir, f"{run_name}_preds.jsonl"),
    )

    # Construir JSON final según Sección 2.7
    results_record = {
        "run": run_name,
        "cfg": cfg,
        "params": n_params,
        "flops_per_token": flops,
        "train_time_s": train_time_s,
        "curve": curve,
        "eval": eval_results,
    }

    results_path = os.path.join(results_dir, f"{run_name}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_record, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"  Resultados guardados en: {results_path}")
        for split in ("test_id", "test_tpl", "test_ood"):
            if split in eval_results:
                r = eval_results[split]
                print(f"  {split}: por nivel {r['acc_by_level']} | total {r['acc_total']} | por tipo {r['acc_by_type']}")

    return results_record, model


def sanity_check_overfit(
    tokenizer: Optional[CharTokenizer] = None,
    device: Optional[torch.device] = None,
    num_samples: int = 32,
    max_steps: int = 300,
    loss_threshold: float = 0.05,
    verbose: bool = True,
) -> bool:
    """Prueba de cordura (Sección 3, paso 3): sobreajustar 32 ejemplos.

    La pérdida tiene que llegar casi a 0 (< 0.05). Si no, hay un bug en los datos o en la máscara.
    """
    if device is None:
        device = get_device()
    if tokenizer is None:
        tokenizer = get_tokenizer()

    set_seed(42)

    # Generamos 32 ejemplos de prueba
    from src.data import generate_toy_problem

    samples = [generate_toy_problem(level=1, prob_id=f"sanity_{i}", seed=100 + i) for i in range(num_samples)]
    dataset = MathDataset(samples, tokenizer=tokenizer)
    loader = create_dataloader(dataset, tokenizer=tokenizer, batch_size=num_samples, shuffle=False)

    # Modelo pequeño para convergencia inmediata
    cfg = {
        "arch": "gpt",
        "n_layers": 2,
        "emb_dim": 128,
        "n_heads": 4,
        "vocab_size": tokenizer.vocab_size,
        "context_length": 512,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    model.train()
    batch = next(iter(loader))
    x = batch["x"].to(device)
    y = batch["y"].to(device)

    final_loss = 999.0
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg["vocab_size"]), y.view(-1), ignore_index=IGNORE_INDEX)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if final_loss < loss_threshold:
            if verbose:
                print(f"✓ Prueba de sobreajuste superada en el paso {step}: loss = {final_loss:.5f} < {loss_threshold}")
            return True

        if verbose and (step % 50 == 0 or step == max_steps - 1):
            print(f"  [Sanity Check] Paso {step:3d}: Loss = {final_loss:.4f}")

    if final_loss < loss_threshold:
        return True

    raise AssertionError(
        f"La prueba de sobreajuste falló: loss final = {final_loss:.4f} >= {loss_threshold}. "
        "Comprueba la máscara de la pérdida o los datos."
    )


if __name__ == "__main__":
    print("Ejecutando prueba de cordura (sobreajuste en 32 ejemplos)...")
    sanity_check_overfit()
