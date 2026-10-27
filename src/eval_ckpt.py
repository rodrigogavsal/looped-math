"""Evalúa un checkpoint ya entrenado y escribe results/<run>.json (sin reentrenar).

Sirve cuando se cortó un run después de guardar checkpoints/<run>.pt pero antes de
terminar la evaluación (el lanzador ignora el .pt y reentrenaría desde cero).

Uso:
  python -m src.eval_ckpt --run looped_N8_d256_s0
  python -m src.eval_ckpt --run looped_N8_d256_s0 --log results/terminal_s0.log

La curva de validación y el tiempo de entrenamiento no están en el .pt: se recuperan del
log de la terminal (si se pasa --log); si no, `curve` queda vacío y `train_time_s` a null.
Después, `python -m src.run ...` salta ese run porque ya existe el .json.
"""
import argparse
import json
import os
import re

import torch

from src.evaluate import run_full_evaluation
from src.model import build_model, count_params, flops_per_token
from src.tokenizer import get_tokenizer
from src.train import get_device

STEP_RE = re.compile(r"Paso\s+(\d+)/\d+ \| Train Loss: ([\d.]+) \| Val CE: ([\d.]+) \| "
                     r"Val Acc \(TF\): ([\d.]+)% \| LR: \S+ \| Tiempo: ([\d.]+)s")


def curve_from_log(log_path, run_name):
    """Última ejecución de `run_name` en el log: (curva, tiempo de entrenamiento en s)."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith(f"[{run_name}] Modelo")]
    if not starts:
        return [], None
    curve, last_time = [], None
    for l in lines[starts[-1] + 1:]:
        if "Checkpoint guardado" in l:
            break
        m = STEP_RE.search(l)
        if m:
            curve.append({"step": int(m[1]), "train_loss": float(m[2]), "val_ce_answer": float(m[3]),
                          "val_acc": float(m[4]) / 100})
            last_time = float(m[5])
    return curve, last_time


def main():
    ap = argparse.ArgumentParser(description="Evalúa un checkpoint y escribe results/<run>.json")
    ap.add_argument("--run", required=True, help="p. ej. looped_N8_d256_s0")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--log", default=None, help="log de la terminal para recuperar curva y tiempo")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = os.path.join("results", f"{a.run}.json")
    if os.path.exists(out) and not a.force:
        raise SystemExit(f"Ya existe {out}; usa --force para sobrescribir.")
    ckpt_path = os.path.join("checkpoints", f"{a.run}.pt")
    dev = get_device() if a.device is None else torch.device(a.device)
    ckpt = torch.load(ckpt_path, map_location=dev)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint cargado: {ckpt_path} (paso {ckpt.get('step')})")

    curve, train_time = curve_from_log(a.log, a.run) if a.log else ([], None)
    d = a.data_dir
    ev = run_full_evaluation(
        model=model,
        test_id_source=os.path.join(d, "test_id.jsonl"),
        test_ood_source=os.path.join(d, "test_ood.jsonl"),
        tokenizer=get_tokenizer(), device=dev,
        run_loops_sweep=(cfg.get("arch", "looped") == "looped"),
        test_tpl_source=os.path.join(d, "test_tpl.jsonl"),
        preds_path=os.path.join("results", f"{a.run}_preds.jsonl"),
    )
    record = {"run": a.run, "cfg": cfg, "params": count_params(model), "flops_per_token": flops_per_token(cfg),
              "train_time_s": train_time, "curve": curve, "eval": ev}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en {out}")
    for split in ("test_id", "test_tpl", "test_ood"):
        if split in ev:
            print(f"  {split}: por nivel {ev[split]['acc_by_level']} | total {ev[split]['acc_total']}")


if __name__ == "__main__":
    main()
