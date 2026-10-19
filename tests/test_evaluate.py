"""Pruebas unitarias para el módulo de evaluación (Parte 3)."""

import pytest
import torch
from src.evaluate import evaluate_generation, evaluate_loops_sweep, extract_number
from src.model import build_model
from src.tokenizer import get_tokenizer


def test_extract_number():
    assert extract_number("15") == 15
    assert extract_number(" 42 ") == 42
    assert extract_number("Respuesta: 100") == 100
    assert extract_number("El resultado es -5") == -5
    assert extract_number("No hay número aquí") is None
    assert extract_number("") is None


def test_evaluate_generation_and_sweep():
    tok = get_tokenizer()
    cfg = {
        "arch": "looped",
        "n_loops": 2,
        "emb_dim": 64,
        "n_heads": 4,
        "vocab_size": tok.vocab_size,
        "context_length": 128,
    }
    model = build_model(cfg)
    device = torch.device("cpu")

    items = [
        {"id": "t1", "level": 1, "question": "Ana tiene 2 canicas. Gana 1. ¿Cuántas tiene?", "answer": 3},
        {"id": "t2", "level": 2, "question": "Luis tiene 5 manzanas. Pierde 2. ¿Cuántas tiene?", "answer": 3},
    ]

    acc_by_level, records = evaluate_generation(model, items, tok, device, max_new_tokens=4)
    assert "1" in acc_by_level
    assert "2" in acc_by_level
    assert len(records) == 2

    # Barrido de loops para N=2 debe evaluar loops 1..4 (1..2N)
    sweep = evaluate_loops_sweep(model, items, tok, device)
    assert set(sweep.keys()) == {"1", "2", "3", "4"}


def test_full_evaluation_with_tpl_types_and_preds(tmp_path):
    import json
    from src.evaluate import run_full_evaluation
    tok = get_tokenizer()
    model = build_model({"arch": "looped", "n_loops": 2, "emb_dim": 32, "n_heads": 4,
                         "vocab_size": tok.vocab_size, "context_length": 128})
    def item(i, lvl, typ="A", sh=False):
        return {"id": f"x{i}", "level": lvl, "type": typ, "shuffled": sh, "template": "A|t",
                "question": "Ana tiene 2 canicas. ¿Cuántas tiene?", "answer": 2}
    test_id = [item(1, 1), item(2, 2, "B", True)]
    test_tpl = [item(3, 1, "B", False)]
    test_ood = [item(4, 5), item(5, 6)]
    preds = tmp_path / "preds.jsonl"
    out = run_full_evaluation(model, test_id, test_ood, tok, torch.device("cpu"),
                              test_tpl_source=test_tpl, preds_path=str(preds))
    assert set(out) == {"test_id", "test_tpl", "test_ood", "loops_sweep", "loops_sweep_ood"}
    assert set(out["test_id"]["acc_by_type"]) == {"A", "B desordenado"}
    assert set(out["loops_sweep_ood"]["1"]) == {"5", "6"}
    rows = [json.loads(l) for l in preds.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5 and {r["split"] for r in rows} == {"test_id", "test_tpl", "test_ood"}
