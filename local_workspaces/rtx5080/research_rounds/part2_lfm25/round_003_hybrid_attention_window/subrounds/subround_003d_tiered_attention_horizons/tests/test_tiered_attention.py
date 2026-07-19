from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULL_LAYERS = [2, 5, 8, 10, 12, 14]
SAFE_MAP = "2:4096,5:4096,8:8192,10:8192,12:8192,14:8192"


def load_overlay(*, enabled: bool = True, window_map: str = SAFE_MAP):
    os.environ["VLLM_R003_WINDOW"] = "1" if enabled else "0"
    os.environ["R003_ATTENTION_WINDOW"] = "4096"
    os.environ["R003D_WINDOW_MAP"] = window_map
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
    path = ROOT / "overlay" / "tiered_attention.py"
    name = f"r003d_test_overlay_{abs(hash((enabled, window_map)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def test_safe_map_routes_every_full_attention_layer():
    module = load_overlay()
    fake = build_fake_module(module)
    instance = fake.Lfm2ForCausalLM()
    expected = {2: 4096, 5: 4096, 8: 8192, 10: 8192, 12: 8192, 14: 8192}
    observed = {
        index: instance.model.layers[index].self_attn.attn.sliding_window
        for index in FULL_LAYERS
    }
    assert observed == expected


def test_map_must_cover_exact_full_attention_topology():
    try:
        load_overlay(window_map="2:4096,5:4096")
    except ValueError as exc:
        assert "cover exactly" in str(exc)
    else:
        raise AssertionError("partial tiered map was accepted")


def test_duplicate_layer_fails_during_parse():
    try:
        load_overlay(
            window_map="2:4096,2:8192,5:4096,8:8192,10:8192,12:8192,14:8192"
        )
    except ValueError as exc:
        assert "invalid" in str(exc)
    else:
        raise AssertionError("duplicate tiered-map layer was accepted")


def test_malformed_prefix_fails_before_attention_init():
    module = load_overlay()
    fake = build_fake_module(module)
    try:
        fake.Attention(prefix="model.bad_prefix")
    except RuntimeError as exc:
        assert "cannot parse" in str(exc)
    else:
        raise AssertionError("malformed prefix was accepted")


def test_disabled_keeps_native_attention():
    module = load_overlay(enabled=False)
    fake = types.SimpleNamespace(Attention=BaseAttention)
    module._patch_lfm2(fake)
    assert fake.r002k is True
    assert fake.Attention is BaseAttention


if __name__ == "__main__":
    test_safe_map_routes_every_full_attention_layer()
    test_map_must_cover_exact_full_attention_topology()
    test_duplicate_layer_fails_during_parse()
    test_malformed_prefix_fails_before_attention_init()
    test_disabled_keeps_native_attention()
    print("r003d_tiered_attention_tests=passed")
