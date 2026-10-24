"""Lanzador de experimentos según la Sección 2.6 y Parte 3 (paso 5) del contrato común.

Uso:
  python -m src.run --runs looped_N4 gpt_L4 --seeds 0 1
  python -m src.run --all --seeds 0
  python -m src.run --runs looped_N4_d256 --seeds 0 --steps 200 (para pruebas rápidas)

Características:
- Carga configs/base.yaml y configs/experiments.yaml
- Salta automáticamente los runs que ya tengan su results/<run>.json (reanudable)
- Escribe logs tanto en consola como en results/experiments.log
- Almacena checkpoints/<run>.pt y results/<run>.json con el formato estricto del contrato
"""

import argparse
import datetime
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.data import create_toy_dataset_splits
from src.evaluate import print_eval_samples
from src.tokenizer import get_tokenizer
from src.train import get_device, train_run


def setup_logger(log_file: str = "results/experiments.log") -> logging.Logger:
    """Configura el logger dual (consola + fichero)."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("looped_experiments")
    logger.setLevel(logging.INFO)

    # Limpiar handlers previos si los hay
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Fichero
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_run_configs(
    base_cfg: Dict[str, Any],
    exp_cfg: Dict[str, Any],
    requested_runs: Optional[List[str]],
    requested_seeds: List[int],
    run_all: bool = False,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Genera la lista de (run_name, full_cfg) resolviendo nombres y semillas."""
    # 1. Aplanar base_cfg si contiene secciones 'model' o 'train'
    flat_base = {}
    if "model" in base_cfg and isinstance(base_cfg["model"], dict):
        flat_base.update(base_cfg["model"])
    if "train" in base_cfg and isinstance(base_cfg["train"], dict):
        flat_base.update(base_cfg["train"])
    for k, v in base_cfg.items():
        if k not in ("model", "train"):
            flat_base[k] = v

    # Incorporar 'defaults' de exp_cfg si existen
    if "defaults" in exp_cfg and isinstance(exp_cfg["defaults"], dict):
        flat_base.update(exp_cfg["defaults"])

    # 2. Normalizar available_runs (soporta dict 'runs' o lista 'experiments')
    available_runs = {}
    if "experiments" in exp_cfg and isinstance(exp_cfg["experiments"], list):
        for entry in exp_cfg["experiments"]:
            name = entry.get("name")
            if name:
                available_runs[name] = entry
    elif "runs" in exp_cfg and isinstance(exp_cfg["runs"], dict):
        available_runs = exp_cfg["runs"]

    selected_configs = []

    for run_key, run_params in available_runs.items():
        # Comprobar si coincide con alguno de los patrones solicitados
        match = run_all
        if requested_runs:
            for req in requested_runs:
                if req == run_key or req in run_key or run_key.startswith(req):
                    match = True
                    break

        if match:
            for seed in requested_seeds:
                full_cfg = dict(flat_base)
                full_cfg.update(run_params)
                full_cfg["seed"] = seed

                # Nombre canónico según sección 2.6: e.g. looped_N4_d256_s0
                arch = full_cfg.get("arch", "looped")
                d = full_cfg.get("emb_dim", 256)
                if arch == "gpt":
                    L = full_cfg.get("n_layers", 4)
                    run_name = f"gpt_L{L}_d{d}_s{seed}"
                else:
                    N = full_cfg.get("n_loops", 4)
                    run_name = f"looped_N{N}_d{d}_s{seed}"

                selected_configs.append((run_name, full_cfg))

    return selected_configs


def main():
    parser = argparse.ArgumentParser(description="Lanzador de experimentos Looped Transformer")
    parser.add_argument("--runs", nargs="+", help="Nombres o prefijos de runs a ejecutar (e.g. looped_N4 gpt_L4)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Semillas a ejecutar (e.g. 0 1 2)")
    parser.add_argument("--all", action="store_true", help="Ejecutar todas las configuraciones de la matriz")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Ruta a experiments.yaml")
    parser.add_argument("--base-config", default="configs/base.yaml", help="Ruta a base.yaml")
    parser.add_argument("--data-dir", default="data", help="Directorio con los ficheros JSONL")
    parser.add_argument("--steps", type=int, default=None, help="Sobrescribir el número de pasos de entrenamiento")
    parser.add_argument("--eval-every", type=int, default=None, help="Sobrescribir la frecuencia de evaluación en val")
    parser.add_argument("--batch-size", type=int, default=None, help="Sobrescribir el tamaño de batch")
    parser.add_argument("--force", action="store_true", help="Forzar re-ejecución incluso si ya existe results/<run>.json")
    parser.add_argument("--demo-samples", type=int, default=10, help="Número de ejemplos a imprimir al final del run")
    parser.add_argument("--device", default=None, help="Dispositivo forzado: cuda, mps, cpu")

    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Iniciando lanzador de experimentos")

    # Asegurar que existan los datos
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")
    test_id_path = os.path.join(args.data_dir, "test_id.jsonl")
    test_ood_path = os.path.join(args.data_dir, "test_ood.jsonl")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_id_path, test_ood_path]):
        logger.warning(f"No se encontraron todos los ficheros de datos en '{args.data_dir}'. Generando datos de juguete...")
        create_toy_dataset_splits(output_dir=args.data_dir)
        logger.info("Datos de juguete creados con éxito.")

    base_cfg = load_yaml(args.base_config)
    exp_cfg = load_yaml(args.config)

    runs_to_execute = resolve_run_configs(
        base_cfg=base_cfg,
        exp_cfg=exp_cfg,
        requested_runs=args.runs,
        requested_seeds=args.seeds,
        run_all=args.all or (not args.runs),
    )

    if not runs_to_execute:
        logger.error(f"No se encontró ninguna configuración coincidente para runs={args.runs}.")
        sys.exit(1)

    logger.info(f"Se ejecutarán {len(runs_to_execute)} runs:")
    for name, _ in runs_to_execute:
        logger.info(f"  - {name}")

    tokenizer = get_tokenizer()
    dev = get_device() if args.device is None else torch.device(args.device)
    logger.info(f"Dispositivo de cómputo seleccionado: {dev}")

    for idx, (run_name, cfg) in enumerate(runs_to_execute, 1):
        results_file = os.path.join("results", f"{run_name}.json")
        if os.path.exists(results_file) and not args.force:
            logger.info(f"[{idx}/{len(runs_to_execute)}] Saltando {run_name} porque ya existe: {results_file}")
            continue

        if args.steps is not None:
            cfg["total_steps"] = args.steps
        if args.eval_every is not None:
            cfg["eval_every"] = args.eval_every
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size

        logger.info(f"[{idx}/{len(runs_to_execute)}] Ejecutando run: {run_name}")

        try:
            record, model = train_run(
                cfg=cfg,
                train_source=train_path,
                val_source=val_path,
                test_id_source=test_id_path,
                test_ood_source=test_ood_path,
                tokenizer=tokenizer,
                device=dev,
                run_name=run_name,
                verbose=True,
            )
            logger.info(f"✓ Run {run_name} completado con éxito en {record['train_time_s']}s.")

            # Muestra cualitativa de generación manual si se solicita
            if args.demo_samples > 0:
                import json
                with open(test_id_path, "r", encoding="utf-8") as f:
                    samples = [json.loads(line) for line in f][: args.demo_samples]
                print_eval_samples(model, samples, tokenizer, dev, num_samples=args.demo_samples)

        except Exception as e:
            logger.exception(f"✗ Error durante la ejecución del run {run_name}: {e}")

    logger.info("Lanzador de experimentos ha finalizado todas las tareas.")


if __name__ == "__main__":
    main()
