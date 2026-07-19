from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULL_LAYERS = [2, 5, 8, 10, 12, 14]


def load_overlay(*, enabled: bool = True, globals_: str = "14"):
    os.environ["VLLM_R003_WINDOW"] = "1" if enabled else "0"
    os.environ["R003_ATTENTION_WINDOW"] = "4096"
    os.environ["R003B_GLOBAL_LAYERS"] = globals_
    base = types.ModuleType("fused_greedy_head")
    base.LFM2_MODULE = "vllm.model_executor.models.lfm2"
    base.SAMPLER_MODULE = "vllm.v1.sample.sampler"
    base.RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
    base.PATCHERS = {
        base.LFM2_MODULE: lambda module: setattr(module, "r002k", True),
        base.SAMPLER_MODULE: lambda module: setattr(module, "r002k", True),
        base.RUNNER_MODULE: lambda module: setattr(module, "r002k", True),
    }
    sys.modules["fused_greedy_head"] = base
    path = ROOT / "overlay" / "selective_global_attention.py"
    spec = importlib.util.spec_from_file_location("r003b_test_overlay", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["r003b_test_overlay"] = module
    spec.loader.exec_module(module)
    return module


class BaseAttention:
    def __init__(self, *args, **kwargs):
        self.sliding_window = kwargs.get("per_layer_sliding_window")


def build_fake_module(module):
    fake = types.SimpleNamespace(Attention=BaseAttention)

    class Model:
        def __init__(self):
            layers = []
            layer_types = []
            for index in range(16):
                if index in FULL_LAYERS:
                    attention = fake.Attention(
                        prefix=f"model.layers.{index}.self_attn.attn"
                    )
                    layers.append(
                        types.SimpleNamespace(
                            self_attn=types.SimpleNamespace(attn=attention)
                        )
                    )
                    layer_types.append("full_attention")
                else:
                    layers.append(types.SimpleNamespace(short_conv=object()))
                    layer_types.append("conv")
            self.model = types.SimpleNamespace(
                layers=layers,
                config=types.SimpleNamespace(layer_types=layer_types),
            )

    fake.Lfm2ForCausalLM = Model
    module._patch_lfm2(fake)
    return fake


def test_final_layer_remains_global():
    module = load_overlay(globals_="14")
    fake = build_fake_module(module)
    instance = fake.Lfm2ForCausalLM()
    for index in FULL_LAYERS:
        observed = instance.model.layers[index].self_attn.attn.sliding_window
        assert observed is None if index == 14 else observed == 4096


def test_first_and_final_layers_remain_global():
    module = load_overlay(globals_="2,14")
    fake = build_fake_module(module)
    instance = fake.Lfm2ForCausalLM()
    assert instance.model.layers[2].self_attn.attn.sliding_window is None
    assert instance.model.layers[14].self_attn.attn.sliding_window is None
    assert instance.model.layers[8].self_attn.attn.sliding_window == 4096


def test_unknown_global_layer_fails_verification():
    module = load_overlay(globals_="3")
    fake = build_fake_module(module)
    try:
        fake.Lfm2ForCausalLM()
    except RuntimeError as exc:
        assert "non-attention global layers" in str(exc)
    else:
        raise AssertionError("non-attention global layer was accepted")


def test_malformed_prefix_fails_before_attention_init():
    module = load_overlay(globals_="14")
    fake = build_fake_module(module)
    try:
        fake.Attention(prefix="model.bad_prefix")
    except RuntimeError as exc:
        assert "cannot parse" in str(exc)
    else:
        raise AssertionError("malformed prefix was accepted")


def test_disabled_keeps_native_attention():
    module = load_overlay(enabled=False, globals_="14")
    fake = types.SimpleNamespace(Attention=BaseAttention)
    module._patch_lfm2(fake)
    assert fake.r002k is True
    assert fake.Attention is BaseAttention


if __name__ == "__main__":
    test_final_layer_remains_global()
    test_first_and_final_layers_remain_global()
    test_unknown_global_layer_fails_verification()
    test_malformed_prefix_fails_before_attention_init()
    test_disabled_keeps_native_attention()
    print("r003b_selective_global_attention_tests=passed")
