"""Apply a 4096 window to selected LFM2 attention layers only."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import re
import sys
from types import ModuleType
from typing import Any

import fused_greedy_head


LFM2_MODULE = fused_greedy_head.LFM2_MODULE
SAMPLER_MODULE = fused_greedy_head.SAMPLER_MODULE
RUNNER_MODULE = fused_greedy_head.RUNNER_MODULE
WINDOW = int(os.environ.get("R003_ATTENTION_WINDOW", "4096"))
WINDOW_ENABLED = os.environ.get("VLLM_R003_WINDOW", "0") == "1"
PROFILE_ENABLED = bool(os.environ.get("R002Y_PROFILE_ARM"))
_LAYER_PREFIX = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.attn$")
_HOOK_INSTALLED = False


def _parse_global_layers(raw: str) -> frozenset[int]:
    if not raw.strip():
        return frozenset()
    pieces = [piece.strip() for piece in raw.split(",")]
    if any(not piece for piece in pieces):
        raise ValueError(f"invalid R003B_GLOBAL_LAYERS={raw!r}")
    values = [int(piece) for piece in pieces]
    if any(value < 0 for value in values) or len(values) != len(set(values)):
        raise ValueError(f"invalid R003B_GLOBAL_LAYERS={raw!r}")
    return frozenset(values)


GLOBAL_LAYERS = _parse_global_layers(os.environ.get("R003B_GLOBAL_LAYERS", ""))


def _layer_index(prefix: Any) -> int:
    if not isinstance(prefix, str):
        raise RuntimeError(f"R003b missing attention prefix: {prefix!r}")
    match = _LAYER_PREFIX.search(prefix)
    if match is None:
        raise RuntimeError(f"R003b cannot parse attention layer prefix: {prefix!r}")
    return int(match.group(1))


def _patch_lfm2(module: ModuleType) -> None:
    """Keep R002k and route full-attention layers by their exact prefix."""

    fused_greedy_head.PATCHERS[LFM2_MODULE](module)
    if not WINDOW_ENABLED:
        return

    attention_base = getattr(module, "Attention", None)
    if attention_base is None:
        raise RuntimeError("R003b could not find LFM2 Attention global")
    if getattr(attention_base, "_r003_window_wrapper", False):
        return

    class SelectiveWindowAttention(attention_base):
        _r003_window_wrapper = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            layer_index = _layer_index(kwargs.get("prefix"))
            existing = kwargs.get("per_layer_sliding_window")
            is_global = layer_index in GLOBAL_LAYERS
            if is_global:
                if existing is not None:
                    raise RuntimeError(
                        "R003b global layer has a pre-existing window: "
                        f"layer={layer_index} window={existing}"
                    )
            else:
                if existing is not None and int(existing) != WINDOW:
                    raise RuntimeError(
                        "R003b found a conflicting per-layer window: "
                        f"layer={layer_index} {existing} != {WINDOW}"
                    )
                kwargs["per_layer_sliding_window"] = WINDOW
            super().__init__(*args, **kwargs)
            observed = getattr(self, "sliding_window", None)
            if is_global and observed is not None:
                raise RuntimeError(
                    f"R003b global layer {layer_index} unexpectedly has {observed}"
                )
            if not is_global and int(observed) != WINDOW:
                raise RuntimeError(
                    f"R003b window was not installed on layer {layer_index}"
                )
            self._r003b_layer_index = layer_index
            self._r003b_is_global = is_global

    module.Attention = SelectiveWindowAttention
    base_model = getattr(module, "Lfm2ForCausalLM", None)
    if base_model is None:
        raise RuntimeError("R003b could not find Lfm2ForCausalLM")
    if getattr(base_model, "_r003b_window_verified", False):
        return

    native_init = base_model.__init__

    @functools.wraps(native_init)
    def init_and_verify(self: Any, *args: Any, **kwargs: Any) -> None:
        native_init(self, *args, **kwargs)
        model = getattr(self, "model", None)
        layers = list(getattr(model, "layers", []))
        layer_types = list(getattr(getattr(model, "config", None), "layer_types", []))
        full_indices = [
            index
            for index, layer_type in enumerate(layer_types)
            if layer_type == "full_attention"
        ]
        unknown_globals = sorted(GLOBAL_LAYERS.difference(full_indices))
        if unknown_globals:
            raise RuntimeError(
                "R003b requested non-attention global layers: "
                f"{unknown_globals}; full={full_indices}"
            )
        expected_global = sorted(GLOBAL_LAYERS)
        expected_windowed = sorted(set(full_indices).difference(GLOBAL_LAYERS))
        observed_global: list[int] = []
        observed_windowed: list[int] = []
        shortconv_indices: list[int] = []
        for index, layer in enumerate(layers):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None:
                attention = getattr(self_attn, "attn", None)
                marker_index = getattr(attention, "_r003b_layer_index", None)
                if marker_index != index:
                    raise RuntimeError(
                        "R003b layer marker mismatch: "
                        f"model={index} attention={marker_index}"
                    )
                observed = getattr(attention, "sliding_window", None)
                if observed is None:
                    observed_global.append(index)
                elif int(observed) == WINDOW:
                    observed_windowed.append(index)
                else:
                    raise RuntimeError(
                        f"R003b unexpected window on layer {index}: {observed}"
                    )
            elif getattr(layer, "short_conv", None) is not None:
                shortconv_indices.append(index)
        if (
            observed_global != expected_global
            or observed_windowed != expected_windowed
            or len(shortconv_indices) != 10
        ):
            raise RuntimeError(
                "R003b layer verification failed: "
                f"expected_global={expected_global} observed_global={observed_global} "
                f"expected_windowed={expected_windowed} "
                f"observed_windowed={observed_windowed} shortconv={shortconv_indices}"
            )
        base_model._r003b_window_verified = True
        print(
            "R003_WINDOW verified=1 "
            f"window={WINDOW} full_layers={observed_windowed} "
            f"global_layers={observed_global} shortconv_layers={shortconv_indices}",
            file=sys.stderr,
            flush=True,
        )

    base_model.__init__ = init_and_verify
    base_model._r003b_window_verified = False
    print(
        "R003_WINDOW lfm2_patched=1 "
        f"enabled=1 window={WINDOW} global_layers={sorted(GLOBAL_LAYERS)}",
        file=sys.stderr,
        flush=True,
    )

def _patch_sampler(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[SAMPLER_MODULE](module)


def _patch_runner(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[RUNNER_MODULE](module)
    if PROFILE_ENABLED:
        from prefill_profiler import _patch_runner as patch_profiler_runner

        patch_profiler_runner(module)


PATCHERS = {
    LFM2_MODULE: _patch_lfm2,
    SAMPLER_MODULE: _patch_sampler,
    RUNNER_MODULE: _patch_runner,
}


class _OverlayLoader(importlib.abc.Loader):
    def __init__(self, original_loader: importlib.abc.Loader, fullname: str) -> None:
        self.original_loader = original_loader
        self.fullname = fullname

    def create_module(self, spec: Any) -> Any:
        create_module = getattr(self.original_loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self.original_loader.exec_module(module)
        PATCHERS[self.fullname](module)


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in PATCHERS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if not isinstance(spec.loader, _OverlayLoader):
            spec.loader = _OverlayLoader(spec.loader, fullname)
        return spec


def install_import_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _OverlayFinder())
    _HOOK_INSTALLED = True
    for fullname, patcher in PATCHERS.items():
        loaded = sys.modules.get(fullname)
        if loaded is not None:
            patcher(loaded)
    print(
        "R003_WINDOW hook_installed=1 "
        f"enabled={int(WINDOW_ENABLED)} window={WINDOW} "
        f"global_layers={sorted(GLOBAL_LAYERS)} profile={int(PROFILE_ENABLED)}",
        file=sys.stderr,
        flush=True,
    )
