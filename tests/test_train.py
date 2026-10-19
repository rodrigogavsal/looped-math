"""Pruebas unitarias para el bucle de entrenamiento y checkpoints (Parte 3)."""

import os
import pytest
import torch
from src.model import build_model
from src.train import lr_at, sanity_check_overfit


def test_lr_schedule():
    base_lr = 1e-3
    warmup = 200
    total = 10000

    # En el paso 0, es base_lr / warmup
    lr_0 = lr_at(0, total, base_lr, warmup)
    assert pytest.approx(lr_0, rel=1e-3) == base_lr / warmup

    # Al finalizar el warmup (paso 199/200), debe ser prácticamente base_lr
    lr_warmup = lr_at(199, total, base_lr, warmup)
    assert pytest.approx(lr_warmup, rel=1e-2) == base_lr

    # Al final de los pasos (paso 10000), decae hasta el 10% del base_lr
    lr_final = lr_at(total, total, base_lr, warmup)
    assert pytest.approx(lr_final, rel=1e-3) == 0.1 * base_lr


def test_sanity_check_overfit():
    # El overfit en 32 ejemplos debe pasar y converger a loss < 0.05
    device = torch.device("cpu")
    success = sanity_check_overfit(device=device, num_samples=16, max_steps=200, verbose=False)
    assert success is True


def test_checkpoint_saving_and_loading(tmp_path):
    cfg = {
        "arch": "looped",
        "n_loops": 2,
        "emb_dim": 64,
        "n_heads": 4,
        "vocab_size": 100,
        "context_length": 64,
    }
    model = build_model(cfg)
    ckpt_file = os.path.join(tmp_path, "test_ckpt.pt")

    # Guardar
    torch.save({"cfg": cfg, "model_state": model.state_dict(), "step": 100}, ckpt_file)
    assert os.path.exists(ckpt_file)

    # Cargar
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 100
    assert ckpt["cfg"]["arch"] == "looped"

    new_model = build_model(ckpt["cfg"])
    new_model.load_state_dict(ckpt["model_state"])

    # Comprobar que los parámetros son idénticos
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
