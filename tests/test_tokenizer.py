"""Pruebas unitarias para el CharTokenizer (Sección 2.3 del contrato común)."""

import pytest
from src.tokenizer import CharTokenizer, get_tokenizer


def test_special_ids():
    tok = get_tokenizer()
    assert tok.pad_id == 0
    assert tok.eos_id == 1
    assert tok.unk_id == 2
    assert tok.vocab_size >= 90  # ~100 símbolos


def test_encode_decode_roundtrip():
    tok = get_tokenizer()
    sample = "Ana tiene 12 canicas. Gana 8 + 3 = 11. ¿Cuántas tiene? ¡Ojo! Niño ágil, pingüino."
    ids = tok.encode(sample)
    assert all(isinstance(i, int) for i in ids)
    assert tok.unk_id not in ids, "Todos los caracteres del texto de prueba deberían estar en el vocabulario"

    decoded = tok.decode(ids, skip_special=True)
    assert decoded == sample


def test_multiplication_sign():
    tok = get_tokenizer()
    # '×' es el signo de multiplicación español oficial
    ids1 = tok.encode("3 × 4")
    # '*' se normaliza a '×' para robustez
    ids2 = tok.encode("3 * 4")
    assert ids1 == ids2
