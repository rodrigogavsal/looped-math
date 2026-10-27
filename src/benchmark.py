"""Benchmark de memoria VRAM, velocidad por paso y tabla de parámetros/FLOPs."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml

from src.model import (
    build_model,
    count_params,
    effective_depth,
    flops_per_token,
)


def _autocast_ctx(device: torch.device, precision: str = "auto"):
    """Contexto de precisión mixta automática (bf16 o fp16 en CUDA, nulo en CPU)."""
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext(), False
    if precision == "bf16" or (
        precision == "auto"
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    ):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16), False
    return torch.autocast(device_type="cuda", dtype=torch.float16), True


def load_experiment_configs(
    repo_root: Path,
    include_optional: bool = True,
) -> List[Dict[str, Any]]:
    """Carga las configuraciones de modelos desde configs/experiments.yaml."""
    exp_path = repo_root / "configs" / "experiments.yaml"
    data = yaml.safe_load(exp_path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    configs: List[Dict[str, Any]] = []
    for entry in data.get("experiments", []):
        if entry.get("optional", False) and not include_optional:
            continue
        configs.append({**defaults, **entry})
    return configs


def build_params_flops_rows(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calcula parámetros, bloques distintos, profundidad efectiva y FLOPs/token."""
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        model = build_model(cfg)
        params = count_params(model)
        flops = flops_per_token(cfg)
        rows.append(
            {
                "name": cfg["name"],
                "arch": "GPT" if cfg["arch"] == "gpt" else "Looped",
                "emb_dim": int(cfg["emb_dim"]),
                "n_heads": int(cfg["n_heads"]),
                "n_layers": int(cfg["n_layers"]),
                "n_loops": int(cfg["n_loops"]),
                "effective_depth": effective_depth(cfg),
                "params": params,
                "params_m": params / 1e6,
                "flops_per_token": flops,
                "flops_m": flops / 1e6,
                "optional": bool(cfg.get("optional", False)),
            }
        )
    return rows


def format_params_flops_markdown(rows: List[Dict[str, Any]]) -> str:
    """Formatea la tabla de parámetros y FLOPs en Markdown."""
    lines = [
        "| Modelo | Arq. | $d$ | Cabezas | Bloques ($L$) | Loops ($N$) | Prof. efectiva | Parámetros | Parámetros (M) | FLOPs / token (M) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        tag = " *(opc.)*" if r["optional"] else ""
        lines.append(
            f"| `{r['name']}`{tag} | {r['arch']} | {r['emb_dim']} | {r['n_heads']} | "
            f"{r['n_layers']} | {r['n_loops']} | {r['effective_depth']} | "
            f"{r['params']:,} | {r['params_m']:.3f} | {r['flops_m']:.2f} |"
        )
    return "\n".join(lines)


def format_params_flops_latex(rows: List[Dict[str, Any]]) -> str:
    """Formatea la tabla de parámetros y FLOPs en LaTeX (booktabs) para el artículo."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Modelo & Arq. & $d$ & Bloques ($L$) & Loops ($N$) & Prof. & Par\'ametros (M) & FLOPs/tok (M) \\",
        r"\midrule",
    ]
    for r in rows:
        name_tex = r["name"].replace("_", r"\_")
        lines.append(
            f"\\texttt{{{name_tex}}} & {r['arch']} & {r['emb_dim']} & "
            f"{r['n_layers']} & {r['n_loops']} & {r['effective_depth']} & "
            f"{r['params_m']:.2f} & {r['flops_m']:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Par\'ametros entrenables y coste computacional (FLOPs por token en el pase \textit{forward} con $T=512$) para las configuraciones de la matriz experimental.}",
            r"\label{tab:params_flops}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def benchmark_single_config(
    cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int = 64,
    seq_len: int = 512,
    warmup_steps: int = 2,
    timed_steps: int = 5,
    total_train_steps: int = 42000,
    precision: str = "auto",
    grad_accum: int = 1,
    grad_checkpoint: bool = False,
) -> Dict[str, Any]:
    """
    Ejecuta pasos de entrenamiento (forward + backward + clip_grad + AdamW) con datos
    aleatorios y devuelve el pico de VRAM y segundos por paso.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(cfg).to(device)
    model.grad_checkpoint = grad_checkpoint
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1
    )
    amp_ctx, use_scaler = _autocast_ctx(device, precision=precision)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    vocab_size = int(cfg["vocab_size"])
    micro = batch_size // grad_accum
    x = torch.randint(0, vocab_size, (micro, seq_len), device=device)
    y = torch.randint(0, vocab_size, (micro, seq_len), device=device)

    def _one_step() -> None:
        # Igual que train.py: grad_accum trozos de batch_size/grad_accum ejemplos por paso.
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            with amp_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)) / grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    for _ in range(warmup_steps):
        _one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(timed_steps):
        _one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    sec_per_step = elapsed / max(1, timed_steps)
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    est_hours_10k = (sec_per_step * total_train_steps) / 3600.0

    del model, optimizer, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": cfg["name"],
        "arch": cfg["arch"],
        "effective_depth": effective_depth(cfg),
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "grad_checkpoint": grad_checkpoint,
        "seq_len": seq_len,
        "sec_per_step": round(sec_per_step, 4),
        "tokens_per_sec": round((batch_size * seq_len) / sec_per_step, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "est_hours_per_run": round(est_hours_10k, 2),
        "est_hours_3_seeds": round(est_hours_10k * 3, 2),
    }


def format_benchmark_markdown(bench_rows: List[Dict[str, Any]], device_desc: str, total_steps: int = 42000) -> str:
    """Formatea los resultados del benchmark en una tabla Markdown."""
    if not bench_rows:
        return f"**Dispositivo medido**: {device_desc}  \nNinguna configuración cabe en memoria."
    lines = [
        f"**Dispositivo medido**: {device_desc}  ",
        f"**Configuración de paso**: `batch_size = {bench_rows[0]['batch_size']}`, `seq_len = {bench_rows[0]['seq_len']}`",
        "",
        f"| Modelo | Prof. | Memoria | Segundos / paso | Tokens / s | VRAM Pico (MB) | Horas est. (1 run, {total_steps} pasos) | Horas est. (3 semillas) |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    total_1_seed = 0.0
    total_3_seeds = 0.0
    for r in bench_rows:
        total_1_seed += r["est_hours_per_run"]
        total_3_seeds += r["est_hours_3_seeds"]
        vram_str = f"{r['peak_vram_mb']:.1f}" if r["peak_vram_mb"] > 0 else "CPU"
        mem = f"grad_accum={r.get('grad_accum', 1)}" + (" + ckpt" if r.get("grad_checkpoint") else "")
        lines.append(
            f"| `{r['name']}` | {r['effective_depth']} | {mem} | {r['sec_per_step']:.4f} s | "
            f"{r['tokens_per_sec']:,.0f} | {vram_str} | "
            f"{r['est_hours_per_run']:.2f} h | {r['est_hours_3_seeds']:.2f} h |"
        )
    lines.append(
        f"| **TOTAL MATRIZ** | - | - | - | - | - | **{total_1_seed:.2f} h** | **{total_3_seeds:.2f} h** |"
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Genera la tabla de parámetros/FLOPs y ejecuta el benchmark de VRAM y velocidad."
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño de batch (por defecto: 64)")
    parser.add_argument("--seq-len", type=int, default=411,
                        help="Longitud de secuencia (por defecto: 411, el ejemplo más largo del dataset; peor caso real)")
    parser.add_argument("--warmup", type=int, default=1, help="Pasos de calentamiento (por defecto: 1)")
    parser.add_argument("--steps", type=int, default=3, help="Pasos medidos por modelo (por defecto: 3)")
    parser.add_argument("--total-steps", type=int, default=42000, help="Pasos totales de un run para estimar horas")
    parser.add_argument("--skip-benchmark", action="store_true", help="Solo genera la tabla de parámetros y FLOPs")
    parser.add_argument("--no-optional", action="store_true", help="Excluye la configuración opcional gpt_L4_d128")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    figures_dir = repo_root / "figures"
    results_dir = repo_root / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = load_experiment_configs(repo_root, include_optional=not args.no_optional)

    # 1. Tabla de parámetros y FLOPs
    param_rows = build_params_flops_rows(configs)
    md_table = format_params_flops_markdown(param_rows)
    tex_table = format_params_flops_latex(param_rows)

    (figures_dir / "model_params_flops.md").write_text(md_table + "\n", encoding="utf-8")
    (figures_dir / "model_params_flops.tex").write_text(tex_table + "\n", encoding="utf-8")

    print("\n=== TABLA DE PARÁMETROS Y FLOPS (MARKDOWN) ===\n")
    print(md_table)
    print(f"\nGuardada en: {figures_dir / 'model_params_flops.md'}")
    print(f"LaTeX en   : {figures_dir / 'model_params_flops.tex'}")

    if args.skip_benchmark:
        return

    # 2. Benchmark de memoria VRAM y velocidad
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        device_desc = f"{torch.cuda.get_device_name(0)} (CUDA)"
    else:
        device_desc = f"CPU ({platform.processor() or platform.machine()})"

    print(f"\n=== BENCHMARK DE MEMORIA Y VELOCIDAD ({device_desc}) ===")
    print(f"Ejecutando {args.steps} pasos medidos (+{args.warmup} warmup) con batch={args.batch_size} x seq_len={args.seq_len}...\n")

    # Si una configuración no cabe en memoria, se prueba con el batch partido en 2 y 4 trozos
    # (grad_accum) y, por último, con checkpointing de activaciones. El resultado del
    # entrenamiento es el mismo en todos los casos; solo cambian la memoria y la velocidad.
    # La primera opción que cabe es la que hay que poner en configs/experiments.yaml.
    ladder = [(1, False), (2, False), (4, False), (4, True)]
    bench_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        row = None
        for accum, ckpt in ladder:
            try:
                row = benchmark_single_config(
                    cfg=cfg,
                    device=device,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    warmup_steps=args.warmup,
                    timed_steps=args.steps,
                    total_train_steps=args.total_steps,
                    grad_accum=accum,
                    grad_checkpoint=ckpt,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"  .. {cfg['name']:<16} no cabe con grad_accum={accum}"
                      f"{' + checkpointing' if ckpt else ''}; probando la siguiente opción")
        if row is None:
            print(f"  -> {cfg['name']:<16} | NO CABE ni con grad_accum=4 + checkpointing")
            continue
        bench_rows.append(row)
        extra = f"grad_accum={row['grad_accum']}" + (" + grad_checkpoint" if row["grad_checkpoint"] else "")
        print(
            f"  -> {row['name']:<16} | {row['sec_per_step']:.4f} s/step | "
            f"VRAM: {row['peak_vram_mb']:.1f} MB | {extra} | "
            f"{args.total_steps} pasos: {row['est_hours_per_run']:.2f} h"
        )

    bench_md = format_benchmark_markdown(bench_rows, device_desc=device_desc, total_steps=args.total_steps)
    (figures_dir / "benchmark_vram_speed.md").write_text(bench_md + "\n", encoding="utf-8")
    (results_dir / "benchmark_vram_speed.json").write_text(
        json.dumps({"device": device_desc, "rows": bench_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== RESUMEN DEL BENCHMARK ===\n")
    print(bench_md)
    print(f"\nGuardado en: {figures_dir / 'benchmark_vram_speed.md'} y {results_dir / 'benchmark_vram_speed.json'}")


if __name__ == "__main__":
    main()
