from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_overlay(monkeypatch, *, enabled: bool):
    monkeypatch.setenv("VLLM_R003_WINDOW", "1" if enabled else "0")
    monkeypatch.setenv("R003_ATTENTION_WINDOW", "8192")
    base = types.ModuleType("fused_greedy_head")
    base.LFM2_MODULE = "vllm.model_executor.models.lfm2"
    base.SAMPLER_MODULE = "vllm.v1.sample.sampler"
    base.RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
    base.PATCHERS = {
        base.LFM2_MODULE: lambda module: setattr(module, "r002k", True),
        base.SAMPLER_MODULE: lambda module: setattr(module, "r002k", True),
        base.RUNNER_MODULE: lambda module: setattr(module, "r002k", True),
    }
    monkeypatch.setitem(sys.modules, "fused_greedy_head", base)
    path = ROOT / "overlay" / "windowed_attention.py"
    spec = importlib.util.spec_from_file_location("r003_test_overlay", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["r003_test_overlay"] = module
    spec.loader.exec_module(module)
    return module


def test_disabled_keeps_base_attention(monkeypatch):
    module = load_overlay(monkeypatch, enabled=False)
    fake = types.SimpleNamespace(Attention=type("Attention", (), {}))
    module._patch_lfm2(fake)
    assert fake.r002k is True
    assert fake.Attention.__name__ == "Attention"


def test_window_wrapper_injects_per_layer_window(monkeypatch):
    module = load_overlay(monkeypatch, enabled=True)

    class BaseAttention:
        def __init__(self, *args, **kwargs):
            self.sliding_window = kwargs.get("per_layer_sliding_window")

    class Model:
        def __init__(self, *args, **kwargs):
            self.model = types.SimpleNamespace(
                layers=[
                    types.SimpleNamespace(self_attn=types.SimpleNamespace(attn=BaseAttention(per_layer_sliding_window=8192)))
                    for _ in range(6)
                ]
                + [types.SimpleNamespace(short_conv=object()) for _ in range(10)],
                config=types.SimpleNamespace(layer_types=["full_attention"] * 6 + ["conv"] * 10),
            )

    fake = types.SimpleNamespace(Attention=BaseAttention, Lfm2ForCausalLM=Model)
    module._patch_lfm2(fake)
    wrapped = fake.Attention()
    assert wrapped.sliding_window == 8192
    instance = fake.Lfm2ForCausalLM()
    assert instance.model.layers[0].self_attn.attn.sliding_window == 8192


def test_conflicting_existing_window_is_rejected(monkeypatch):
    module = load_overlay(monkeypatch, enabled=True)

    class BaseAttention:
        def __init__(self, *args, **kwargs):
            pass

    fake = types.SimpleNamespace(
        Attention=BaseAttention,
        Lfm2ForCausalLM=type("Model", (), {"__init__": lambda self: None}),
    )
    module._patch_lfm2(fake)
    try:
        fake.Attention(per_layer_sliding_window=4096)
    except RuntimeError as exc:
        assert "conflicting" in str(exc)
    else:
        raise AssertionError("conflicting window was accepted")


class _MonkeyPatch:
    def setenv(self, key, value):
        import os

        os.environ[key] = value

    def setitem(self, mapping, key, value):
        mapping[key] = value


if __name__ == "__main__":
    test_disabled_keeps_base_attention(_MonkeyPatch())
    test_window_wrapper_injects_per_layer_window(_MonkeyPatch())
    test_conflicting_existing_window_is_rejected(_MonkeyPatch())
    print("r003_window_overlay_tests=passed")
