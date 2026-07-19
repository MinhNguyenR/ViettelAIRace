"""Apply an explicit per-layer attention-window map to LFM2 full attention."""

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
WINDOW_ENABLED = os.environ.get("VLLM_R003_WINDOW", "0") == "1"
PRIMARY_WINDOW = int(os.environ.get("R003_ATTENTION_WINDOW", "4096"))
PROFILE_ENABLED = bool(os.environ.get("R002Y_PROFILE_ARM"))
FULL_LAYERS = frozenset({2, 5, 8, 10, 12, 14})
_LAYER_PREFIX = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.attn$")
_HOOK_INSTALLED = False


def _parse_window_map(raw: str) -> dict[int, int]:
    if not raw.strip():
        raise ValueError("R003D_WINDOW_MAP is required")
    result: dict[int, int] = {}
    for item in raw.split(","):
        pieces = item.strip().split(":")
        if len(pieces) != 2:
            raise ValueError(f"invalid R003D_WINDOW_MAP item: {item!r}")
        layer, window = (int(piece.strip()) for piece in pieces)
        if layer in result or window <= 0:
            raise ValueError(f"invalid R003D_WINDOW_MAP item: {item!r}")
        result[layer] = window
    if set(result) != FULL_LAYERS:
        raise ValueError(
            "R003D_WINDOW_MAP must cover exactly the six full-attention layers: "
            f"observed={sorted(result)} expected={sorted(FULL_LAYERS)}"
        )
    return result


WINDOW_MAP = _parse_window_map(
    os.environ.get(
        "R003D_WINDOW_MAP",
        "2:4096,5:4096,8:8192,10:8192,12:8192,14:8192",
    )
)


def _layer_index(prefix: Any) -> int:
    if not isinstance(prefix, str):
        raise RuntimeError(f"R003d missing attention prefix: {prefix!r}")
    match = _LAYER_PREFIX.search(prefix)
    if match is None:
        raise RuntimeError(f"R003d cannot parse attention layer prefix: {prefix!r}")
    return int(match.group(1))


def _patch_lfm2(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[LFM2_MODULE](module)
    if not WINDOW_ENABLED:
        return

    attention_base = getattr(module, "Attention", None)
    if attention_base is None:
        raise RuntimeError("R003d could not find LFM2 Attention global")
    if getattr(attention_base, "_r003d_tiered_wrapper", False):
        return

    class TieredWindowAttention(attention_base):
        _r003d_tiered_wrapper = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            layer_index = _layer_index(kwargs.get("prefix"))
            if layer_index not in WINDOW_MAP:
                raise RuntimeError(
                    f"R003d received unexpected attention layer {layer_index}"
                )
            window = WINDOW_MAP[layer_index]
            existing = kwargs.get("per_layer_sliding_window")
            if existing is not None and int(existing) != window:
                raise RuntimeError(
                    "R003d found a conflicting per-layer window: "
                    f"layer={layer_index} {existing} != {window}"
                )
            kwargs["per_layer_sliding_window"] = window
            super().__init__(*args, **kwargs)
            observed = getattr(self, "sliding_window", None)
            if int(observed) != window:
                raise RuntimeError(
                    f"R003d window was not installed on layer {layer_index}"
                )
            self._r003d_layer_index = layer_index
            self._r003d_window = window

    module.Attention = TieredWindowAttention
    base_model = getattr(module, "Lfm2ForCausalLM", None)
    if base_model is None:
        raise RuntimeError("R003d could not find Lfm2ForCausalLM")
    if getattr(base_model, "_r003d_window_verified", False):
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
        if set(full_indices) != FULL_LAYERS:
            raise RuntimeError(
                f"R003d unexpected full-attention topology: {full_indices}"
            )
        observed_map: dict[int, int] = {}
        shortconv_indices: list[int] = []
        for index, layer in enumerate(layers):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None:
                attention = getattr(self_attn, "attn", None)
                marker_index = getattr(attention, "_r003d_layer_index", None)
                if marker_index != index:
                    raise RuntimeError(
                        "R003d layer marker mismatch: "
                        f"model={index} attention={marker_index}"
                    )
                observed_map[index] = int(
                    getattr(attention, "sliding_window", -1)
                )
            elif getattr(layer, "short_conv", None) is not None:
                shortconv_indices.append(index)
        if observed_map != WINDOW_MAP or len(shortconv_indices) != 10:
            raise RuntimeError(
                "R003d layer verification failed: "
                f"expected={WINDOW_MAP} observed={observed_map} "
                f"shortconv={shortconv_indices}"
            )
        base_model._r003d_window_verified = True
        map_text = ",".join(
            f"{layer}:{WINDOW_MAP[layer]}" for layer in sorted(WINDOW_MAP)
        )
        print(
            "R003_WINDOW verified=1 "
            f"window={PRIMARY_WINDOW} tiered_map={map_text} "
            f"shortconv_layers={shortconv_indices}",
            file=sys.stderr,
            flush=True,
        )

    base_model.__init__ = init_and_verify
    base_model._r003d_window_verified = False
    map_text = ",".join(
        f"{layer}:{WINDOW_MAP[layer]}" for layer in sorted(WINDOW_MAP)
    )
    print(
        "R003_WINDOW lfm2_patched=1 "
        f"enabled=1 window={PRIMARY_WINDOW} tiered_map={map_text}",
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
    map_text = ",".join(
        f"{layer}:{WINDOW_MAP[layer]}" for layer in sorted(WINDOW_MAP)
    )
    print(
        "R003_WINDOW hook_installed=1 "
        f"enabled={int(WINDOW_ENABLED)} window={PRIMARY_WINDOW} "
        f"tiered_map={map_text} profile={int(PROFILE_ENABLED)}",
        file=sys.stderr,
        flush=True,
    )
