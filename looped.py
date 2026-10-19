"""Script principal de conveniencia para la Práctica 1: Looped Transformer.

Permite interactuar fácilmente con el pipeline de entrenamiento, evaluación y pruebas de cordura:
  - python looped.py --sanity-check
  - python looped.py --train --run looped_N4_d256 --seed 0 --steps 1000
  - python looped.py --eval --checkpoint checkpoints/looped_N4_d256_s0.pt
  - python looped.py --demo
"""

import argparse
import json
import os
import sys
import torch

from src.data import create_dataloader, create_toy_dataset_splits
from src.evaluate import print_eval_samples, run_full_evaluation
from src.model import build_model
from src.tokenizer import get_tokenizer
from src.train import get_device, sanity_check_overfit, train_run


def main():
    parser = argparse.ArgumentParser(description="Looped Transformer CLI")
    parser.add_argument("--sanity-check", action="store_true", help="Ejecutar prueba de sobreajuste en 32 ejemplos")
    parser.add_argument("--train", action="store_true", help="Entrenar un run")
    parser.add_argument("--run", default="looped_N4_d256", help="Configuración de run (e.g. looped_N4_d256)")
    parser.add_argument("--seed", type=int, default=0, help="Semilla aleatoria")
    parser.add_argument("--steps", type=int, default=None, help="Número de pasos")
    parser.add_argument("--eval", action="store_true", help="Evaluar un checkpoint existente")
    parser.add_argument("--checkpoint", default=None, help="Ruta al checkpoint .pt")
    parser.add_argument("--demo", action="store_true", help="Imprimir 10 ejemplos generados con un checkpoint")
    parser.add_argument("--data-dir", default="data", help="Directorio de datos JSONL")

    args = parser.parse_args()
    device = get_device()
    tokenizer = get_tokenizer()

    if args.sanity_check:
        print(f"Ejecutando prueba de cordura en {device}...")
        sanity_check_overfit(tokenizer=tokenizer, device=device)
        return

    if args.demo or args.eval:
        ckpt_path = args.checkpoint or f"checkpoints/{args.run}_s{args.seed}.pt"
        if not os.path.exists(ckpt_path):
            print(f"Error: No se encontró el checkpoint {ckpt_path}")
            sys.exit(1)

        print(f"Cargando checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(ckpt["cfg"]).to(device)
        model.load_state_dict(ckpt["model_state"])

        test_id_path = os.path.join(args.data_dir, "test_id.jsonl")
        test_ood_path = os.path.join(args.data_dir, "test_ood.jsonl")

        if args.demo:
            with open(test_id_path, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f][:10]
            print_eval_samples(model, items, tokenizer, device, num_samples=10)

        if args.eval:
            print("Ejecutando evaluación completa...")
            results = run_full_evaluation(model, test_id_path, test_ood_path, tokenizer, device)
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.train:
        # Importar y delegar en run.py
        from src.run import load_yaml, resolve_run_configs
        base_cfg = load_yaml("configs/base.yaml")
        exp_cfg = load_yaml("configs/experiments.yaml")
        selected = resolve_run_configs(base_cfg, exp_cfg, [args.run], [args.seed])
        if not selected:
            print(f"No se encontró la configuración {args.run}")
            sys.exit(1)
        run_name, cfg = selected[0]
        if args.steps:
            cfg["total_steps"] = args.steps
        train_run(
            cfg=cfg,
            train_source=os.path.join(args.data_dir, "train.jsonl"),
            val_source=os.path.join(args.data_dir, "val.jsonl"),
            test_id_source=os.path.join(args.data_dir, "test_id.jsonl"),
            test_ood_source=os.path.join(args.data_dir, "test_ood.jsonl"),
            tokenizer=tokenizer,
            device=device,
            run_name=run_name,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
