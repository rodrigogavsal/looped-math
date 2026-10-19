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

    # Barrido de loops para N=2 debe evaluar loops 1..2 (1..N)
    sweep = evaluate_loops_sweep(model, items, tok, device)
    assert set(sweep.keys()) == {"1", "2"}
