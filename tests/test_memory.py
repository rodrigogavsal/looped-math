"""Los ahorros de memoria no cambian el resultado.

- Atención eficiente (SDPA) frente a la explícita del notebook: mismos logits.
- GELU fusionada frente a la fórmula del notebook: mismo valor.
- Checkpointing de activaciones: mismos gradientes.
- Acumulación de gradiente (grad_accum): mismo gradiente que un único batch.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.data import IGNORE_INDEX
from src.model import DEFAULT_MODEL_CFG, GELU, build_model


def _cfg(arch, n_layers=2, n_loops=3):
    return {**DEFAULT_MODEL_CFG, "arch": arch, "context_length": 64, "emb_dim": 64, "n_heads": 4,
            "n_layers": n_layers, "n_unique_layers": 1, "n_loops": n_loops}


@pytest.mark.parametrize("arch", ["gpt", "looped"])
def test_sdpa_equals_explicit_attention(arch):
    torch.manual_seed(0)
    model = build_model(_cfg(arch)).eval()
    idx = torch.randint(0, DEFAULT_MODEL_CFG["vocab_size"], (3, 40))
    with torch.no_grad():
        fast = model(idx)                              # SDPA
        slow, _ = model(idx, return_attn=True)         # explícita del notebook
    assert torch.allclose(fast, slow, atol=1e-5), (fast - slow).abs().max()


def test_fused_gelu_equals_notebook_formula():
    x = torch.linspace(-6, 6, 1001)
    ref = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    assert torch.allclose(GELU()(x), ref, atol=1e-6)


@pytest.mark.parametrize("arch", ["gpt", "looped"])
def test_grad_checkpoint_same_gradients(arch):
    torch.manual_seed(1)
    model = build_model(_cfg(arch)).train()
    idx = torch.randint(0, DEFAULT_MODEL_CFG["vocab_size"], (2, 30))

    def grads(flag):
        model.grad_checkpoint = flag
        model.zero_grad()
        model(idx).float().pow(2).mean().backward()
        return [p.grad.clone() for p in model.parameters()]

    for a, b in zip(grads(False), grads(True)):
        assert torch.allclose(a, b, atol=1e-6)


def test_grad_accum_same_gradient_as_full_batch():
    """Partir el batch en trozos y normalizar por el total de tokens da el mismo gradiente."""
    torch.manual_seed(2)
    model = build_model(_cfg("gpt")).train()
    V = DEFAULT_MODEL_CFG["vocab_size"]
    x = torch.randint(3, V, (8, 20))
    y = x.clone()
    y[:, :12] = IGNORE_INDEX                     # solo los últimos tokens cuentan, como en los datos
    y[::2, 17:] = IGNORE_INDEX                   # distinto nº de tokens válidos por ejemplo

    model.zero_grad()
    logits = model(x)
    F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=IGNORE_INDEX).backward()
    full = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    n_valid = int((y != IGNORE_INDEX).sum())
    for xs, ys in ((x[:3], y[:3]), (x[3:], y[3:])):
        logits = model(xs)
        (F.cross_entropy(logits.view(-1, V), ys.view(-1), ignore_index=IGNORE_INDEX, reduction="sum")
         / n_valid).backward()
    accum = [p.grad.clone() for p in model.parameters()]

    for a, b in zip(full, accum):
        assert torch.allclose(a, b, atol=1e-6)
