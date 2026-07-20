"""Install R003f split range with an exact-native R003g short path."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType
from typing import Any

import fused_greedy_head
from policy import FULL_LAYERS, effective_split_prefix_tokens, parse_window_map
from split_range_unified_attention import unified_attention as split_unified_attention


LFM2_MODULE = fused_greedy_head.LFM2_MODULE
SAMPLER_MODULE = fused_greedy_head.SAMPLER_MODULE
RUNNER_MODULE = fused_greedy_head.RUNNER_MODULE
TRITON_BACKEND_MODULE = "vllm.v1.attention.backends.triton_attn"
ENABLED = os.environ.get("R003F_ENABLED", "1") == "1"
WINDOW_MAP = parse_window_map(
    os.environ.get(
        "R003F_WINDOW_MAP",
        "2:4096,5:4096,8:4096,10:4096,12:4096,14:4096",
    )
)
PREFIX_TOKENS = int(os.environ.get("R003F_PREFIX_TOKENS", "128"))
if PREFIX_TOKENS < 0:
    raise ValueError("R003F_PREFIX_TOKENS must be non-negative")
SHORT_NATIVE_ENABLED = os.environ.get("R003G_SHORT_NATIVE", "1") == "1"

_LAYER_MARKER = "_r003f_split_range"
_RUNNER_MARKER = "_r003f_split_range_runner"
_BACKEND_MARKER = "_r003f_split_range_backend"
_ANNOUNCED_KERNEL = False
_ANNOUNCED_SHORT_NATIVE = False


def _layer_index(prefix: Any) -> int:
    if not isinstance(prefix, str):
        raise RuntimeError(f"R003f missing attention prefix: {prefix!r}")
    marker = ".layers."
    if marker not in prefix or ".self_attn.attn" not in prefix:
        raise RuntimeError(f"R003f cannot parse attention prefix: {prefix!r}")
    return int(prefix.split(marker, 1)[1].split(".", 1)[0])


def _patch_triton_backend(module: ModuleType) -> None:
    if not ENABLED or getattr(module, _BACKEND_MARKER, False):
        return
    native_unified_attention = getattr(module, "unified_attention", None)
    if native_unified_attention is None:
        raise RuntimeError("R003f Triton backend has no unified_attention symbol")

    @functools.wraps(native_unified_attention)
    def run_split_range(*args: Any, **kwargs: Any) -> Any:
        global _ANNOUNCED_KERNEL, _ANNOUNCED_SHORT_NATIVE
        if PREFIX_TOKENS == 0:
            return native_unified_attention(*args, **kwargs)
        if kwargs.get("mm_prefix_range") is not None:
            raise RuntimeError("R003f refuses native mm-prefix metadata")
        max_seqlen_k = kwargs.get("max_seqlen_k")
        window_size = kwargs.get("window_size")
        if not isinstance(max_seqlen_k, int):
            raise RuntimeError(
                f"R003g expected integer max_seqlen_k, got {max_seqlen_k!r}"
            )
        if not isinstance(window_size, tuple):
            raise RuntimeError(f"R003g expected tuple window_size, got {window_size!r}")
        effective_prefix_tokens = effective_split_prefix_tokens(
            max_seqlen_k=max_seqlen_k,
            window_size=window_size,
            configured_prefix_tokens=PREFIX_TOKENS,
            short_native_enabled=SHORT_NATIVE_ENABLED,
        )
        if effective_prefix_tokens == 0:
            if not _ANNOUNCED_SHORT_NATIVE:
                _ANNOUNCED_SHORT_NATIVE = True
                print(
                    "R003G_SHORT_NATIVE active=1 specialization=split_zero "
                    f"max_seqlen_k={max_seqlen_k} window_size={window_size}",
                    file=sys.stderr,
                    flush=True,
                )
            return split_unified_attention(
                *args,
                **kwargs,
                split_prefix_tokens=0,
            )
        if not _ANNOUNCED_KERNEL:
            _ANNOUNCED_KERNEL = True
            print(
                "R003F_KERNEL active=1 compact_tiles=1 prune_before_kv=1 "
                f"prefix_tokens={PREFIX_TOKENS} decode_layout=adaptive_2d_3d",
                file=sys.stderr,
                flush=True,
            )
        return split_unified_attention(
            *args,
            **kwargs,
            split_prefix_tokens=effective_prefix_tokens,
        )

    module.unified_attention = run_split_range
    setattr(module, _BACKEND_MARKER, True)
    print(
        "R003F backend_patched=1 "
        f"prefix_tokens={PREFIX_TOKENS} zero_control={int(PREFIX_TOKENS == 0)} "
        f"short_native={int(SHORT_NATIVE_ENABLED)}",
        file=sys.stderr,
        flush=True,
    )


def _patch_lfm2(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[LFM2_MODULE](module)
    if not ENABLED:
        return
    attention_base = getattr(module, "Attention", None)
    if attention_base is None:
        raise RuntimeError("R003f could not find LFM2 Attention global")
    if getattr(attention_base, _LAYER_MARKER, False):
        return

    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    class SplitRangeAttention(attention_base):
        _r003f_split_range = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            layer_index = _layer_index(kwargs.get("prefix"))
            if layer_index not in WINDOW_MAP:
                raise RuntimeError(f"R003f unexpected attention layer {layer_index}")
            existing = kwargs.get("per_layer_sliding_window")
            if existing is not None and int(existing) != WINDOW_MAP[layer_index]:
                raise RuntimeError(
                    f"R003f conflicting window layer={layer_index} "
                    f"{existing} != {WINDOW_MAP[layer_index]}"
                )
            existing_backend = kwargs.get("attn_backend")
            if existing_backend is not None and existing_backend is not TritonAttentionBackend:
                raise RuntimeError("R003f refuses a non-Triton preselected backend")
            kwargs["per_layer_sliding_window"] = WINDOW_MAP[layer_index]
            kwargs["attn_backend"] = TritonAttentionBackend
            super().__init__(*args, **kwargs)
            if getattr(self.attn_backend, "get_name")() != "TRITON_ATTN":
                raise RuntimeError("R003f Triton backend was not installed")
            if bool(getattr(self, "use_mm_prefix", False)):
                raise RuntimeError("R003f model unexpectedly enabled native mm-prefix")
            self._r003f_layer_index = layer_index
            self._r003f_window = WINDOW_MAP[layer_index]

    module.Attention = SplitRangeAttention
    base_model = getattr(module, "Lfm2ForCausalLM", None)
    if base_model is None:
        raise RuntimeError("R003f could not find Lfm2ForCausalLM")
    if getattr(base_model, "_r003f_window_verified", False):
        return

    native_init = base_model.__init__

    @functools.wraps(native_init)
    def init_and_verify(self: Any, *args: Any, **kwargs: Any) -> None:
        native_init(self, *args, **kwargs)
        model = getattr(self, "model", None)
        layers = list(getattr(model, "layers", []))
        observed: dict[int, tuple[int, str]] = {}
        shortconv: list[int] = []
        for index, layer in enumerate(layers):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None:
                attention = getattr(self_attn, "attn", None)
                marker = getattr(attention, "_r003f_layer_index", None)
                if marker != index:
                    raise RuntimeError(
                        f"R003f layer marker mismatch model={index} attention={marker}"
                    )
                observed[index] = (
                    int(getattr(attention, "_r003f_window", -1)),
                    str(getattr(attention.attn_backend, "get_name")()),
                )
            elif getattr(layer, "short_conv", None) is not None:
                shortconv.append(index)
        expected = {index: (WINDOW_MAP[index], "TRITON_ATTN") for index in FULL_LAYERS}
        if observed != expected or len(shortconv) != 10:
            raise RuntimeError(
                f"R003f topology verification failed expected={expected} "
                f"observed={observed} shortconv={shortconv}"
            )
        if not getattr(sys.modules[TRITON_BACKEND_MODULE], _BACKEND_MARKER, False):
            raise RuntimeError("R003f backend patch was not installed")
        base_model._r003f_window_verified = True
        map_text = ",".join(f"{i}:{WINDOW_MAP[i]}" for i in sorted(WINDOW_MAP))
        print(
            "R003_WINDOW verified=1 "
            f"window={min(WINDOW_MAP.values())} tiered_map={map_text} "
            f"backend=TRITON_ATTN shortconv_layers={shortconv}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "R003F verified=1 attention_layers=6 shortconv_layers=10 "
            f"prefix_tokens={PREFIX_TOKENS} backend=TRITON_ATTN",
            file=sys.stderr,
            flush=True,
        )

    base_model.__init__ = init_and_verify
    base_model._r003f_window_verified = False


def _patch_sampler(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[SAMPLER_MODULE](module)


def _patch_runner(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[RUNNER_MODULE](module)
    if os.environ.get("R002Y_PROFILE_ARM"):
        from prefill_profiler import _patch_runner as patch_profiler_runner

        patch_profiler_runner(module)
    base = getattr(module, "GPUModelRunner", None)
    if base is None or getattr(base, _RUNNER_MARKER, False) or not ENABLED:
        return
    setattr(base, _RUNNER_MARKER, True)
    print("R003F runner_patched=1 prefix_metadata=0", file=sys.stderr, flush=True)


PATCHERS = {
    LFM2_MODULE: _patch_lfm2,
    SAMPLER_MODULE: _patch_sampler,
    RUNNER_MODULE: _patch_runner,
    TRITON_BACKEND_MODULE: _patch_triton_backend,
}


class _OverlayLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, fullname: str) -> None:
        self.loader = loader
        self.fullname = fullname

    def create_module(self, spec: Any) -> Any:
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)
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
    sys.meta_path.insert(0, _OverlayFinder())
    for fullname, patcher in PATCHERS.items():
        loaded = sys.modules.get(fullname)
        if loaded is not None:
            patcher(loaded)
    map_text = ",".join(f"{i}:{WINDOW_MAP[i]}" for i in sorted(WINDOW_MAP))
    print(
        "R003_WINDOW hook_installed=1 enabled=1 "
        f"window={min(WINDOW_MAP.values())} tiered_map={map_text} "
        f"backend=TRITON_ATTN prefix_tokens={PREFIX_TOKENS}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "R003F hook_installed=1 compact_tiles=1 "
        f"prefix_tokens={PREFIX_TOKENS} map={map_text} "
        f"short_native={int(SHORT_NATIVE_ENABLED)}",
        file=sys.stderr,
        flush=True,
    )
