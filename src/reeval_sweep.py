"""Amplía el barrido de loops de test_id (N+1 .. 2N) en runs looped ya entrenados, sin reentrenar.

Contexto: hasta ahora `loops_sweep` (test_id) llegaba solo a N y `loops_sweep_ood` a 2N. Con el cambio de
`src/evaluate.py`, los runs nuevos ya salen con 1..2N en los dos. Este script pone al día los runs
antiguos: carga checkpoints/<run>.pt, evalúa en test_id solo los loops que faltan (N+1..2N) y los
añade a results/<run>.json. Todo lo demás del .json queda igual (se guarda antes una copia .bak).

Antes de añadir nada comprueba que reproduce el valor ya guardado para N loops (si no coincide, avisa y
no toca el fichero).

Uso (desde la raíz del repositorio, con el entorno activado):
    python -m src.reeval_sweep --seeds 0 1            # todos los looped_* de esas semillas
    python -m src.reeval_sweep --runs looped_N8_d256_s0
Es idempotente: si el .json ya tiene 1..2N, lo salta.
"""
import argparse
import glob
import json
import os
import shutil
import time

import torch

from src.evaluate import evaluate_generation
from src.model import build_model
from src.tokenizer import get_tokenizer
from src.train import get_device


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def discover(seeds):
    runs = []
    for s in seeds:
        for path in sorted(glob.glob(os.path.join("results", f"looped_N*_d*_s{s}.json"))):
            runs.append(os.path.splitext(os.path.basename(path))[0])
    return runs


def main():
    ap = argparse.ArgumentParser(description="Amplía loops_sweep de test_id a 1..2N sin reentrenar.")
    ap.add_argument("--runs", nargs="+", help="p. ej. looped_N8_d256_s0")
    ap.add_argument("--seeds", nargs="+", type=int, help="todos los looped_* de estas semillas")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-verify", action="store_true", help="no comprobar el valor ya guardado para N loops")
    a = ap.parse_args()

    runs = list(a.runs or []) + (discover(a.seeds) if a.seeds else [])
    if not runs:
        raise SystemExit("Indica --runs o --seeds.")
    dev = get_device() if a.device is None else torch.device(a.device)
    tok = get_tokenizer()
    items = load_items(os.path.join(a.data_dir, "test_id.jsonl"))
    print(f"{len(runs)} runs: {', '.join(runs)}  (dispositivo {dev})")

    for run in runs:
        t0 = time.time()
        jpath = os.path.join("results", f"{run}.json")
        ppath = os.path.join("checkpoints", f"{run}.pt")
        if not (os.path.exists(jpath) and os.path.exists(ppath)):
            print(f"[{run}] falta {jpath} o {ppath}: se salta")
            continue
        with open(jpath, "r", encoding="utf-8") as f:
            rec = json.load(f)
        if rec["cfg"].get("arch") != "looped":
            print(f"[{run}] no es looped: se salta")
            continue
        n = int(rec["cfg"]["n_loops"])
        sweep = rec["eval"].get("loops_sweep", {})
        missing = [t for t in range(n + 1, 2 * n + 1) if str(t) not in sweep]
        if not missing:
            print(f"[{run}] ya tiene 1..{2 * n}: nada que hacer")
            continue

        ckpt = torch.load(ppath, map_location=dev)
        model = build_model(ckpt["cfg"]).to(dev)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        if not a.no_verify:
            acc_n, _ = evaluate_generation(model, items, tok, dev, n_loops=n)
            old = sweep.get(str(n))
            if old is None or any(abs(acc_n[k] - old[k]) > 1e-9 for k in old):
                print(f"[{run}] ¡NO coincide el valor guardado para {n} loops con el recalculado!\n"
                      f"   guardado:   {old}\n   recalculado: {acc_n}\n   No se modifica el fichero.")
                continue

        new = {}
        for t in missing:
            acc, _ = evaluate_generation(model, items, tok, dev, n_loops=t)
            new[str(t)] = acc
            print(f"[{run}] loops={t}: {acc}")

        shutil.copyfile(jpath, jpath + ".bak")
        merged = dict(sweep)
        merged.update(new)
        rec["eval"]["loops_sweep"] = {k: merged[k] for k in sorted(merged, key=int)}
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print(f"[{run}] actualizado {jpath} (copia en .bak) en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
