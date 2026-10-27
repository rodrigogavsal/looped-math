"""Verificación automática del dataset (Parte 1).

    python -m pytest tests/test_data.py -q                      # sobre un dataset pequeño generado al vuelo
    DATA_DIR=data python -m pytest tests/test_data.py -q        # sobre el dataset real ya generado

Comprueba que cada respuesta sale de su traza y también del texto (resolvedor
independiente), que el nivel es el número de operaciones, que no hay operaciones
seguidas que se anulen, que todo está dentro del vocabulario y de la longitud
máxima, que no hay preguntas repetidas entre ficheros, que test_tpl usa
redacciones reservadas y el resto no, y que la generación es determinista.
"""
import json
import os
import sys
from collections import Counter

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.generate_data import (HELD_IDS, MAX_VAL, MIN_VAL, SPLITS, eval_trace,  # noqa: E402
                               generate_dataset, is_inverse)
from src.text_solver import check_rows  # noqa: E402
from src.tokenizer import (CHARS, MAX_LEN, CharTokenizer, example_length,  # noqa: E402
                           format_example, format_prompt)

FIELDS = {"id", "level", "type", "shuffled", "style", "question", "answer", "trace", "template"}
LEVELS = {name: set(levels) for name, levels, _, _ in SPLITS}
SMALL = os.environ.get("DATA_DIR") is None


def load(d, name):
    with open(os.path.join(d, f"{name}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = os.environ.get("DATA_DIR")
    if d is None:
        d = str(tmp_path_factory.mktemp("data"))
        generate_dataset(d, n_train=400, n_val=60, n_test=80, seed=7, verbose=False)
    return {name: load(d, name) for name, *_ in SPLITS}


def all_rows(data):
    return [r for rows in data.values() for r in rows]


# ------------------------------------------------------------------ esquema y aritmética
def test_schema(data):
    for r in all_rows(data):
        assert set(r) == FIELDS, r["id"]
        assert r["type"] in ("A", "B") and r["style"] in ("compact", "verbose")
        assert isinstance(r["answer"], int)


def test_answer_matches_trace(data):
    for r in all_rows(data):
        result, values = eval_trace(r["trace"])
        assert result == r["answer"], r["id"]
        assert all(MIN_VAL <= v <= MAX_VAL for v in values), (r["id"], values)


def test_level_is_number_of_operations(data):
    for r in all_rows(data):
        assert (len(r["trace"]) - 1) // 2 == r["level"], r["id"]


def test_levels_per_split(data):
    for name, rows in data.items():
        assert {r["level"] for r in rows} == LEVELS[name], name


def test_order_matters_from_level_2(data):
    """A partir del nivel 2 hay al menos una operación multiplicativa (doble, mitad...)."""
    for r in all_rows(data):
        if r["level"] >= 2:
            assert any(op in ("*", "/") for op in r["trace"][1::2]), r["id"]


def test_no_cancelling_consecutive_operations(data):
    """Nunca hay dos operaciones seguidas que se anulen (doble y luego mitad, +k y luego -k)."""
    for r in all_rows(data):
        ops = list(zip(r["trace"][1::2], r["trace"][2::2]))
        for prev, cur in zip(ops, ops[1:]):
            assert not is_inverse(prev, cur), (r["id"], r["trace"])


def test_text_solver_agrees(data):
    """Resolviendo SOLO a partir del texto (src/text_solver.py) sale la misma respuesta y el mismo nº de operaciones."""
    for name, rows in data.items():
        bad = check_rows(rows)
        assert not bad, (name, bad[:5])


def test_files_use_lf_line_endings(tmp_path):
    generate_dataset(str(tmp_path), n_train=10, n_val=2, n_test=2, seed=1, verbose=False)
    for name, *_ in SPLITS:
        assert b"\r" not in (tmp_path / f"{name}.jsonl").read_bytes(), name


def test_numbers_in_question_are_operands(data):
    """Todos los números de la traza (salvo factores implícitos como doble/mitad) aparecen en el texto."""
    for r in all_rows(data):
        nums = {int(t) for t in "".join(c if c.isdigit() else " " for c in r["question"]).split()}
        tr = r["trace"]
        assert tr[0] in nums, r["id"]
        for sym, arg in zip(tr[1::2], tr[2::2]):
            if sym in "+-":
                assert arg in nums, (r["id"], arg)


# ------------------------------------------------------------------ texto y longitud
def test_vocabulary(data):
    tok = CharTokenizer()
    for r in all_rows(data):
        text = format_example(r["question"], r["answer"])
        assert tok.is_supported(text), (r["id"], tok.unsupported_chars(text))


def test_max_length(data):
    for r in all_rows(data):
        assert example_length(r["question"], r["answer"]) <= MAX_LEN, r["id"]


def test_eval_never_longer_than_train(data):
    train_max = max(example_length(r["question"], r["answer"]) for r in data["train"])
    for name, rows in data.items():
        if name != "train":
            assert max(example_length(r["question"], r["answer"]) for r in rows) <= train_max, name


def test_question_format(data):
    for r in all_rows(data):
        q = r["question"]
        assert q.endswith("?") and "¿" in q, r["id"]
        assert "  " not in q and q == q.strip(), r["id"]
        assert q[0].isupper(), r["id"]


# ------------------------------------------------------------------ particiones
def test_no_overlap_between_splits(data):
    seen = {}
    for name, rows in data.items():
        for r in rows:
            assert r["question"] not in seen, (r["id"], seen.get(r["question"]))
            seen[r["question"]] = r["id"]


def test_ids_unique(data):
    ids = [r["id"] for r in all_rows(data)]
    assert len(ids) == len(set(ids))


def test_held_templates_only_in_test_tpl(data):
    held = set(HELD_IDS)
    for name, rows in data.items():
        for r in rows:
            used = set(r["template"].split("|")[1:])
            if name == "test_tpl":
                assert used & held, r["id"]
            else:
                assert not (used & held), (name, r["id"], used & held)


def test_balance(data):
    for name, rows in data.items():
        for level in LEVELS[name]:
            rs = [r for r in rows if r["level"] == level]
            assert Counter(r["type"] for r in rs)["A"] == len(rs) // 2 + len(rs) % 2, (name, level)
    b = [r for r in data["train"] if r["type"] == "B"]
    frac = sum(r["shuffled"] for r in b) / len(b)
    assert abs(frac - 0.5) < (0.1 if SMALL else 0.02), frac


def test_only_b_can_be_shuffled(data):
    for r in all_rows(data):
        if r["type"] == "A":
            assert r["shuffled"] is False


# ------------------------------------------------------------------ determinismo y tokenizador
def test_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_dataset(str(a), n_train=30, n_val=5, n_test=5, seed=3, write_card=False, verbose=False)
    generate_dataset(str(b), n_train=30, n_val=5, n_test=5, seed=3, write_card=False, verbose=False)
    for name, *_ in SPLITS:
        assert (a / f"{name}.jsonl").read_text(encoding="utf-8") == (b / f"{name}.jsonl").read_text(encoding="utf-8")


def test_tokenizer():
    tok = CharTokenizer()
    assert (tok.pad_id, tok.eos_id, tok.unk_id) == (0, 1, 2)
    assert tok.vocab_size == 3 + len(CHARS) == len(set(tok.itos))
    text = format_example("Ana tiene 12 canicas. ¿Cuántas canicas tiene Ana?", 12)
    ids = tok.encode(text, add_eos=True)
    assert ids[-1] == tok.eos_id and len(ids) == example_length("Ana tiene 12 canicas. ¿Cuántas canicas tiene Ana?", 12)
    assert tok.decode(ids) == text
    assert format_prompt("x").endswith("Respuesta: ")
    assert tok.encode("ç") == [tok.unk_id]
