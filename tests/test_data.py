"""Pruebas unitarias para el dataset y dataloader (Sección 2.2 y Parte 3)."""

import pytest
import torch
from src.data import IGNORE_INDEX, MathCollate, MathDataset, create_dataloader, generate_toy_problem
from src.tokenizer import get_tokenizer


def test_dataset_item_masking():
    tok = get_tokenizer()
    sample = {
        "id": "test_1",
        "level": 1,
        "question": "Ana tiene 5 canicas. Gana 3 canicas. ¿Cuántas canicas tiene Ana?",
        "answer": 8,
        "trace": [5, "+", 3],
        "template": "toy",
    }
    dataset = MathDataset([sample], tokenizer=tok)
    item = dataset[0]

    x = item["x"]
    y = item["y"]
    assert len(x) == len(y)

    # Comprobar que en y los tokens de prompt son IGNORE_INDEX (-100)
    prompt_len = item["prompt_len"]
    # Los tokens del prompt (excepto el último que predice la respuesta) deben ser -100
    for idx in range(prompt_len - 1):
        assert y[idx].item() == IGNORE_INDEX

    # Los tokens correspondientes a la respuesta ("8" y <eos>) deben ser != -100
    eval_targets = [val.item() for val in y if val.item() != IGNORE_INDEX]
    assert len(eval_targets) == len(str(sample["answer"])) + 1  # Dígitos de respuesta + 1 (<eos>)
    assert eval_targets[-1] == tok.eos_id


def test_collate_dynamic_padding():
    tok = get_tokenizer()
    samples = [
        generate_toy_problem(level=1, prob_id="s1", seed=1),
        generate_toy_problem(level=4, prob_id="s2", seed=2),
    ]
    dataset = MathDataset(samples, tokenizer=tok)
    loader = create_dataloader(dataset, tokenizer=tok, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    x = batch["x"]
    y = batch["y"]
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape == y.shape
    assert x.shape[0] == 2

    # El elemento más corto debe haber sido rellenado con pad_id en x y con IGNORE_INDEX en y
    len0 = len(dataset[0]["x"])
    len1 = len(dataset[1]["x"])
    min_len = min(len0, len1)
    max_len = max(len0, len1)
    short_idx = 0 if len0 < len1 else 1

    if min_len < max_len:
        assert (x[short_idx, min_len:] == tok.pad_id).all()
        assert (y[short_idx, min_len:] == IGNORE_INDEX).all()
