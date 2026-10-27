"""Tests de corrección de GPTModel y LoopedGPT."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import yaml

from src.model import (
    DEFAULT_MODEL_CFG,
    build_model,
    count_params,
    flops_per_token,
    looped_from_gpt,
    unroll_to_gpt,
)


def _small_cfg(arch: str = "looped", n_layers: int = 1, n_loops: int = 4) -> Dict[str, Any]:
    """Configuración ligera para tests rápidos y deterministas."""
    return {
        **DEFAULT_MODEL_CFG,
        "arch": arch,
        "vocab_size": DEFAULT_MODEL_CFG["vocab_size"],
        "context_length": 64,
        "emb_dim": 64,
        "n_heads": 4,
        "n_layers": n_layers,
        "n_unique_layers": 1,
        "n_loops": n_loops,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }


def test_looped_n1_equals_gpt_l1() -> None:
    """1. Looped N = 1 da exactamente los mismos logits que GPT L = 1 con los mismos pesos."""
    torch.manual_seed(42)
    V = DEFAULT_MODEL_CFG["vocab_size"]
    looped = build_model(_small_cfg(arch="looped", n_layers=1, n_loops=1)).eval()
    gpt = build_model(_small_cfg(arch="gpt", n_layers=1, n_loops=1)).eval()
    gpt.load_state_dict(looped.state_dict())

    idx = torch.randint(0, V, (3, 16))
    with torch.no_grad():
        logits_looped = looped(idx)
        logits_gpt = gpt(idx)

    assert torch.allclose(logits_looped, logits_gpt, atol=1e-6)


@pytest.mark.parametrize("n_loops", [2, 4, 8, 12])
def test_looped_n_equals_unrolled_gpt_n(n_loops: int) -> None:
    """
    2. Looped N da los mismos logits que un GPT de N capas con N copias del bloque
    (pesos compartidos = copias apiladas) para N = 2, 4, 8, 12.
    """
    torch.manual_seed(100 + n_loops)
    V = DEFAULT_MODEL_CFG["vocab_size"]
    looped = build_model(_small_cfg(arch="looped", n_layers=1, n_loops=n_loops)).eval()
    gpt_unrolled = unroll_to_gpt(looped, n_loops=n_loops).eval()

    idx = torch.randint(0, V, (2, 20))
    with torch.no_grad():
        logits_looped, extras_looped = looped(idx, return_hidden=True, return_attn=True)
        logits_gpt, extras_gpt = gpt_unrolled(idx, return_hidden=True, return_attn=True)

    assert torch.allclose(logits_looped, logits_gpt, atol=1e-5)
    assert len(extras_looped["hidden"]) == n_loops + 1
    assert len(extras_gpt["hidden"]) == n_loops + 1
    for h_l, h_g in zip(extras_looped["hidden"], extras_gpt["hidden"]):
        assert torch.allclose(h_l, h_g, atol=1e-5)


def test_looped_params_invariant_to_n_loops() -> None:
    """3. El número de parámetros de LoopedGPT no cambia al variar N."""
    params_by_n = [
        count_params(build_model(_small_cfg(arch="looped", n_layers=1, n_loops=n)))
        for n in (1, 2, 4, 8, 12)
    ]
    assert len(set(params_by_n)) == 1


@pytest.mark.parametrize("arch", ["gpt", "looped"])
def test_causality_mask(arch: str) -> None:
    """4. Causalidad: cambiar tokens futuros no altera los logits de las posiciones anteriores."""
    torch.manual_seed(7)
    V = DEFAULT_MODEL_CFG["vocab_size"]
    cfg = _small_cfg(arch=arch, n_layers=2 if arch == "gpt" else 1, n_loops=4)
    model = build_model(cfg).eval()

    idx_a = torch.randint(0, V, (2, 16))
    idx_b = idx_a.clone()
    cutoff = 8
    idx_b[:, cutoff:] = (idx_b[:, cutoff:] + 17) % V

    with torch.no_grad():
        logits_a = model(idx_a)
        logits_b = model(idx_b)

    assert torch.allclose(logits_a[:, :cutoff, :], logits_b[:, :cutoff, :], atol=1e-6)


@pytest.mark.parametrize("arch", ["gpt", "looped"])
def test_return_hidden_and_attn_do_not_change_logits(arch: str) -> None:
    """5. Pedir return_hidden o return_attn no cambia los logits devueltos."""
    torch.manual_seed(19)
    V = DEFAULT_MODEL_CFG["vocab_size"]
    cfg = _small_cfg(arch=arch, n_layers=3 if arch == "gpt" else 1, n_loops=4)
    model = build_model(cfg).eval()

    idx = torch.randint(0, V, (2, 14))
    with torch.no_grad():
        logits_plain = model(idx)
        logits_h, extras_h = model(idx, return_hidden=True, return_attn=False)
        logits_a, extras_a = model(idx, return_hidden=False, return_attn=True)
        logits_both, extras_both = model(idx, return_hidden=True, return_attn=True)

    assert torch.allclose(logits_plain, logits_h, atol=1e-6)
    assert torch.allclose(logits_plain, logits_a, atol=1e-6)
    assert torch.allclose(logits_plain, logits_both, atol=1e-6)
    assert "hidden" in extras_h and "attn" not in extras_h
    assert "attn" in extras_a and "hidden" not in extras_a
    assert "hidden" in extras_both and "attn" in extras_both


@pytest.mark.parametrize("arch", ["gpt", "looped"])
def test_head_on_last_hidden_equals_final_logits(arch: str) -> None:
    """6. model.head(hidden[-1]) es exactamente igual a los logits finales."""
    torch.manual_seed(23)
    V = DEFAULT_MODEL_CFG["vocab_size"]
    cfg = _small_cfg(arch=arch, n_layers=3 if arch == "gpt" else 1, n_loops=4)
    model = build_model(cfg).eval()

    idx = torch.randint(0, V, (2, 12))
    with torch.no_grad():
        logits, extras = model(idx, return_hidden=True)
        logits_from_head = model.head(extras["hidden"][-1])

    assert torch.allclose(logits, logits_from_head, atol=1e-6)


@pytest.mark.parametrize("arch", ["gpt", "looped"])
def test_attention_rows_sum_to_one_and_zero_above_diagonal(arch: str) -> None:
    """7. Cada fila de atención suma 1 y vale 0 por encima de la diagonal principal."""
    torch.manual_seed(31)
    V = DEFAULT_MODEL_CFG["vocab_size"]
    cfg = _small_cfg(arch=arch, n_layers=2 if arch == "gpt" else 1, n_loops=4)
    model = build_model(cfg).eval()

    seq_len = 15
    idx = torch.randint(0, V, (2, seq_len))
    with torch.no_grad():
        _, extras = model(idx, return_attn=True)

    expected_steps = 2 if arch == "gpt" else 4
    assert len(extras["attn"]) == expected_steps

    upper_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    for attn in extras["attn"]:
        assert attn.shape == (2, cfg["n_heads"], seq_len, seq_len)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        above_diag = attn[:, :, upper_mask]
        assert torch.allclose(above_diag, torch.zeros_like(above_diag), atol=1e-7)


def test_build_all_experiment_matrix_configs() -> None:
    """8. Verifica que build_model construye todas las configuraciones de configs/experiments.yaml."""
    exp_path = Path(__file__).resolve().parent.parent / "configs" / "experiments.yaml"
    data = yaml.safe_load(exp_path.read_text(encoding="utf-8"))
    defaults = data["defaults"]
    experiments = data["experiments"]

    assert len([e for e in experiments if not e.get("optional", False)]) == 8

    idx = torch.randint(0, defaults["vocab_size"], (2, 8))
    for entry in experiments:
        cfg = {**defaults, **entry}
        model = build_model(cfg).eval()
        assert count_params(model) > 0
        assert flops_per_token(cfg) > 0
        with torch.no_grad():
            logits = model(idx)
        assert logits.shape == (2, 8, cfg["vocab_size"])


@pytest.mark.parametrize("mode", ["first", "last", "middle", "mean", "stride"])
def test_looped_from_gpt_modes(mode: str) -> None:
    """9. Verifica la conversión looped_from_gpt en todos sus modos de inicialización."""
    torch.manual_seed(55)
    gpt = build_model(_small_cfg(arch="gpt", n_layers=4, n_loops=1)).eval()
    looped = looped_from_gpt(gpt, n_loops=4, mode=mode).eval()
    assert looped.n_loops == 4
    assert looped.n_unique_layers == 1
