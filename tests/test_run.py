"""Selección de runs del lanzador (src/run.py)."""

from src.run import load_yaml, resolve_run_configs

BASE = load_yaml("configs/base.yaml")
EXP = load_yaml("configs/experiments.yaml")


def names(requested=None, run_all=False, include_optional=False):
    return [n for n, _ in resolve_run_configs(BASE, EXP, requested, [0], run_all=run_all,
                                              include_optional=include_optional)]


def test_prefix_does_not_match_longer_depth():
    assert names(["gpt_L1"]) == ["gpt_L1_d256_s0"]          # no gpt_L12
    assert names(["looped_N8"]) == ["looped_N8_d256_s0"]


def test_optional_only_when_asked():
    assert names(["gpt_L4"]) == ["gpt_L4_d256_s0"]          # no gpt_L4_d128
    assert names(["gpt_L4_d128"]) == ["gpt_L4_d128_s0"]
    assert len(names(run_all=True)) == 9
    assert len(names(run_all=True, include_optional=True)) == 10


def test_steps_and_memory_settings():
    cfg = dict(resolve_run_configs(BASE, EXP, ["gpt_L12"], [0]))["gpt_L12_d256_s0"]
    assert cfg["total_steps"] == 42000 and cfg["batch_size"] == 64 and cfg["grad_accum"] == 2
    cfg = dict(resolve_run_configs(BASE, EXP, ["gpt_L4"], [0]))["gpt_L4_d256_s0"]
    assert cfg["grad_accum"] == 1
