"""Evaluación por generación y teacher forcing según la Sección 2.4, 2.7 y Parte 3 del contrato común.

Maneja:
- Generación greedy token a token hasta <eos> o 6 caracteres como máximo
- Extracción del número generado mediante regex
- Exactitud exacta por nivel (métrica principal)
- Entropía cruzada sobre la respuesta por nivel con teacher forcing (métrica secundaria)
- Agrupamiento por longitud de prompt para generación en batch sin rellenar por la izquierda
- Barrido de loops en test (loops_sweep) para modelos LoopedGPT (1 hasta N)
- Impresión de 10 ejemplos generados para inspección manual
"""

import collections
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.data import IGNORE_INDEX, MathDataset, create_dataloader
from src.tokenizer import CharTokenizer, get_tokenizer


def extract_number(text: str) -> Optional[int]:
    """Extrae el primer número entero de una cadena de texto generada.

    Si no hay número, devuelve None (cuenta como fallo).
    """
    match = re.search(r"-?\d+", text.strip())
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None


@torch.no_grad()
def generate_greedy_batch(
    model: torch.nn.Module,
    prompt_ids_batch: torch.Tensor,
    max_new_tokens: int = 6,
    eos_id: int = 1,
    n_loops: Optional[int] = None,
) -> List[List[int]]:
    """Generación greedy en batch para prompts de EXACTAMENTE la misma longitud.

    Evita el relleno por la izquierda, manteniendo las posiciones absolutas íntegras.
    """
    model.eval()
    b, seq_len = prompt_ids_batch.shape
    device = prompt_ids_batch.device

    idx = prompt_ids_batch.clone()
    done = torch.zeros(b, dtype=torch.bool, device=device)
    generated_tokens: List[List[int]] = [[] for _ in range(b)]

    for _ in range(max_new_tokens):
        # Llamar al modelo (soportando n_loops en LoopedGPT si se pasa)
        if hasattr(model, "n_loops") and n_loops is not None:
            logits = model(idx, n_loops=n_loops)
        else:
            logits = model(idx)

        # Logits del último token: (b, vocab_size)
        next_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_logits, dim=-1)  # (b,)

        # Registrar tokens para las secuencias que no hayan terminado
        for i in range(b):
            if not done[i]:
                tok = next_tokens[i].item()
                if tok == eos_id:
                    done[i] = True
                else:
                    generated_tokens[i].append(tok)

        if done.all():
            break

        idx = torch.cat([idx, next_tokens.unsqueeze(-1)], dim=-1)

    return generated_tokens


@torch.no_grad()
def evaluate_generation(
    model: torch.nn.Module,
    items: List[Dict[str, Any]],
    tokenizer: CharTokenizer,
    device: torch.device,
    n_loops: Optional[int] = None,
    max_new_tokens: int = 6,
    batch_size: int = 64,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evalúa la exactitud exacta por nivel mediante generación greedy.

    Agrupa por longitud de prompt para evaluar en batch sin desajuste posicional.
    Devuelve (acc_by_level, list_of_eval_records).
    """
    model.eval()

    # Preparamos los prompts codificados
    encoded_items = []
    for it in items:
        prompt_text = f"Pregunta: {it['question']}\nRespuesta: "
        p_ids = tokenizer.encode(prompt_text)
        encoded_items.append({
            "item": it,
            "prompt_text": prompt_text,
            "prompt_ids": p_ids,
            "length": len(p_ids),
        })

    # Agrupar por longitud de prompt
    by_length = collections.defaultdict(list)
    for ei in encoded_items:
        by_length[ei["length"]].append(ei)

    results_by_level = collections.defaultdict(list)
    detailed_records = []

    for length, group in by_length.items():
        # Procesar en sub-batches
        for start_idx in range(0, len(group), batch_size):
            chunk = group[start_idx : start_idx + batch_size]
            prompt_tensor = torch.tensor(
                [c["prompt_ids"] for c in chunk], dtype=torch.long, device=device
            )

            gen_tokens_batch = generate_greedy_batch(
                model=model,
                prompt_ids_batch=prompt_tensor,
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_id,
                n_loops=n_loops,
            )

            for c, gen_tokens in zip(chunk, gen_tokens_batch):
                gen_text = tokenizer.decode(gen_tokens, skip_special=True)
                extracted_num = extract_number(gen_text)
                true_ans = int(c["item"]["answer"])
                is_correct = 1.0 if (extracted_num is not None and extracted_num == true_ans) else 0.0

                lvl = str(c["item"].get("level", 1))
                results_by_level[lvl].append(is_correct)

                detailed_records.append({
                    "id": c["item"].get("id"),
                    "level": lvl,
                    "question": c["item"]["question"],
                    "true_answer": true_ans,
                    "generated_text": gen_text,
                    "extracted_number": extracted_num,
                    "correct": is_correct,
                })

    acc_by_level = {
        lvl: round(sum(accs) / len(accs), 4) if accs else 0.0
        for lvl, accs in sorted(results_by_level.items(), key=lambda x: int(x[0]))
    }

    return acc_by_level, detailed_records


@torch.no_grad()
def evaluate_teacher_forcing_ce(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_loops: Optional[int] = None,
) -> Dict[str, float]:
    """Calcula la entropía cruzada media sobre los tokens de la respuesta por nivel."""
    model.eval()
    losses_by_level = collections.defaultdict(list)

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        levels = batch["levels"].tolist()

        if hasattr(model, "n_loops") and n_loops is not None:
            logits = model(x, n_loops=n_loops)
        else:
            logits = model(x)

        # logits: (b, T, vocab_size) -> (b, vocab_size, T)
        logits_t = logits.transpose(1, 2)
        # Pérdida por token individual: (b, T)
        loss_per_token = F.cross_entropy(logits_t, y, ignore_index=IGNORE_INDEX, reduction="none")

        # Para cada elemento del batch, media de pérdida sobre sus tokens evaluados
        valid_mask = y != IGNORE_INDEX
        token_counts = valid_mask.sum(dim=1)  # (b,)
        seq_losses = (loss_per_token * valid_mask).sum(dim=1)  # (b,)

        for i, lvl in enumerate(levels):
            count = token_counts[i].item()
            if count > 0:
                ce = seq_losses[i].item() / count
                losses_by_level[str(lvl)].append(ce)

    ce_by_level = {
        lvl: round(sum(losses) / len(losses), 4) if losses else 0.0
        for lvl, losses in sorted(losses_by_level.items(), key=lambda x: int(x[0]))
    }
    return ce_by_level


@torch.no_grad()
def evaluate_loops_sweep(
    model: torch.nn.Module,
    items: List[Dict[str, Any]],
    tokenizer: CharTokenizer,
    device: torch.device,
    max_loops: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Realiza el barrido de loops (1 hasta N) sobre el conjunto de test.

    Solo aplica a LoopedGPT. Evalúa iteraciones de 1 a N para comprobar si
    con menos loops que los entrenados el modelo ya es capaz de resolver el problema.
    Devuelve {"1": {lvl: acc, ...}, "2": {...}, ...}
    """
    if not hasattr(model, "n_loops"):
        return {}

    n = model.n_loops
    # Límite fijado a N (los loops con los que fue entrenado)
    limit = max_loops if max_loops is not None else n
    sweep_results = {}

    for t in range(1, limit + 1):
        acc_by_lvl, _ = evaluate_generation(
            model=model,
            items=items,
            tokenizer=tokenizer,
            device=device,
            n_loops=t,
        )
        sweep_results[str(t)] = acc_by_lvl

    return sweep_results


def run_full_evaluation(
    model: torch.nn.Module,
    test_id_source: Union[str, List[Dict[str, Any]]],
    test_ood_source: Union[str, List[Dict[str, Any]]],
    tokenizer: CharTokenizer,
    device: torch.device,
    run_loops_sweep: bool = True,
) -> Dict[str, Any]:
    """Ejecuta la batería de evaluación completa según el esquema de la Sección 2.7."""
    # Cargar datos si son rutas
    def _load(src):
        if isinstance(src, str):
            with open(src, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        return list(src)

    test_id_items = _load(test_id_source)
    test_ood_items = _load(test_ood_source)

    # 1. Test ID (niveles 1-4)
    acc_id, _ = evaluate_generation(model, test_id_items, tokenizer, device)
    dl_id = create_dataloader(test_id_items, tokenizer=tokenizer, batch_size=64, shuffle=False)
    ce_id = evaluate_teacher_forcing_ce(model, dl_id, device)

    # 2. Test OOD (niveles 5-6)
    acc_ood, _ = evaluate_generation(model, test_ood_items, tokenizer, device)
    dl_ood = create_dataloader(test_ood_items, tokenizer=tokenizer, batch_size=64, shuffle=False)
    ce_ood = evaluate_teacher_forcing_ce(model, dl_ood, device)

    eval_out = {
        "test_id": {
            "acc_by_level": acc_id,
            "ce_answer_by_level": ce_id,
        },
        "test_ood": {
            "acc_by_level": acc_ood,
            "ce_answer_by_level": ce_ood,
        },
    }

    # 3. Loops sweep si el modelo es LoopedGPT
    if hasattr(model, "n_loops") and run_loops_sweep:
        sweep = evaluate_loops_sweep(model, test_id_items, tokenizer, device)
        eval_out["loops_sweep"] = sweep

    return eval_out


def print_eval_samples(
    model: torch.nn.Module,
    items: List[Dict[str, Any]],
    tokenizer: CharTokenizer,
    device: torch.device,
    num_samples: int = 10,
):
    """Imprime una muestra de preguntas con su respuesta generada y la correcta."""
    sample_subset = items[:num_samples]
    _, records = evaluate_generation(model, sample_subset, tokenizer, device)

    print("=" * 80)
    print(f"MUESTRA DE EVALUACIÓN ({len(records)} ejemplos)")
    print("=" * 80)
    for idx, r in enumerate(records, 1):
        status = "✓ CORRECTO" if r["correct"] == 1.0 else "✗ INCORRECTO"
        print(f"[{idx}] (Nivel {r['level']}) ID: {r['id']}")
        print(f"  Pregunta:  {r['question']}")
        print(f"  Esperado:  {r['true_answer']}")
        print(f"  Generado:  '{r['generated_text']}' -> Extraído: {r['extracted_number']}")
        print(f"  Resultado: {status}")
        print("-" * 80)
