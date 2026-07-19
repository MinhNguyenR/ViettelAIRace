"""Composite R002k/R002y hook plus an opt-in LFM2 attention window.

The window is injected only into the ``Attention`` global used by LFM2's
``Lfm2Attention`` constructor. ShortConv layers never construct that class.
The hook is deliberately fail-fast: a candidate that does not verify all six
full-attention layers is not a valid measurement.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType
from typing import Any

import fused_greedy_head


LFM2_MODULE = fused_greedy_head.LFM2_MODULE
SAMPLER_MODULE = fused_greedy_head.SAMPLER_MODULE
RUNNER_MODULE = fused_greedy_head.RUNNER_MODULE
WINDOW = int(os.environ.get("R003_ATTENTION_WINDOW", "8192"))
WINDOW_ENABLED = os.environ.get("VLLM_R003_WINDOW", "0") == "1"
PROFILE_ENABLED = bool(os.environ.get("R002Y_PROFILE_ARM"))
_HOOK_INSTALLED = False


def _patch_lfm2(module: ModuleType) -> None:
    """Keep R002k's LFM2 patch and optionally inject the six-layer window."""

    fused_greedy_head.PATCHERS[LFM2_MODULE](module)
    if not WINDOW_ENABLED:
        return

    attention_base = getattr(module, "Attention", None)
    if attention_base is None:
        raise RuntimeError("R003 could not find LFM2 Attention global")
    if getattr(attention_base, "_r003_window_wrapper", False):
        return

    class WindowedAttention(attention_base):
        _r003_window_wrapper = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            existing = kwargs.get("per_layer_sliding_window")
            if existing is not None and int(existing) != WINDOW:
                raise RuntimeError(
                    "R003 found a conflicting per-layer attention window: "
                    f"{existing} != {WINDOW}"
                )
            kwargs["per_layer_sliding_window"] = WINDOW
            super().__init__(*args, **kwargs)
            if int(getattr(self, "sliding_window", -1)) != WINDOW:
                raise RuntimeError("R003 attention window was not installed")

    module.Attention = WindowedAttention
    base_model = getattr(module, "Lfm2ForCausalLM", None)
    if base_model is None:
        raise RuntimeError("R003 could not find Lfm2ForCausalLM")
    if getattr(base_model, "_r003_window_verified", False):
        return

    native_init = base_model.__init__

    @functools.wraps(native_init)
    def init_and_verify(self: Any, *args: Any, **kwargs: Any) -> None:
        native_init(self, *args, **kwargs)
        layers = list(getattr(getattr(self, "model", None), "layers", []))
        config = getattr(getattr(self, "model", None), "config", None)
        layer_types = list(getattr(config, "layer_types", []))
        full_indices = [
            index for index, layer_type in enumerate(layer_types)
            if layer_type == "full_attention"
        ]
        observed_window_indices: list[int] = []
        shortconv_indices: list[int] = []
        for index, layer in enumerate(layers):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None:
                attention = getattr(self_attn, "attn", None)
                if int(getattr(attention, "sliding_window", -1)) == WINDOW:
                    observed_window_indices.append(index)
            elif getattr(layer, "short_conv", None) is not None:
                shortconv_indices.append(index)
        if observed_window_indices != full_indices or len(shortconv_indices) != 10:
            raise RuntimeError(
                "R003 layer verification failed: "
                f"expected_full={full_indices} observed_window="
                f"{observed_window_indices} shortconv={shortconv_indices}"
            )
        base_model._r003_window_verified = True
        print(
            "R003_WINDOW verified=1 "
            f"window={WINDOW} full_layers={observed_window_indices} "
            f"shortconv_layers={shortconv_indices}",
            file=sys.stderr,
            flush=True,
        )

    base_model.__init__ = init_and_verify
    base_model._r003_window_verified = False
    print(
        "R003_WINDOW lfm2_patched=1 "
        f"enabled=1 window={WINDOW}",
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
        f"enabled={int(WINDOW_ENABLED)} window={WINDOW} profile={int(PROFILE_ENABLED)}",
        file=sys.stderr,
        flush=True,
    )
