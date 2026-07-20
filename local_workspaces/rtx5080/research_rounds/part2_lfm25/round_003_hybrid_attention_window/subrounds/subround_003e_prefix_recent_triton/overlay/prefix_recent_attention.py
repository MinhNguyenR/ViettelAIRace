"""R003e: native Triton prefix-plus-recent attention overlay for LFM2."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType
from typing import Any

import fused_greedy_head
from policy import FULL_LAYERS, parse_window_map, prefix_ranges


LFM2_MODULE = fused_greedy_head.LFM2_MODULE
SAMPLER_MODULE = fused_greedy_head.SAMPLER_MODULE
RUNNER_MODULE = fused_greedy_head.RUNNER_MODULE
ENABLED = os.environ.get("R003E_ENABLED", "1") == "1"
WINDOW_MAP = parse_window_map(
    os.environ.get(
        "R003E_WINDOW_MAP",
        "2:4096,5:4096,8:4096,10:4096,12:4096,14:4096",
    )
)
PREFIX_TOKENS = int(os.environ.get("R003E_PREFIX_TOKENS", "128"))
if PREFIX_TOKENS < 0:
    raise ValueError("R003E_PREFIX_TOKENS must be non-negative")
PREFIX_ENABLED = PREFIX_TOKENS > 0
_LAYER_MARKER = "_r003e_prefix_recent"
_RUNNER_MARKER = "_r003e_prefix_recent_runner"
_ANNOUNCED_METADATA = False


def _layer_index(prefix: Any) -> int:
    if not isinstance(prefix, str):
        raise RuntimeError(f"R003e missing attention prefix: {prefix!r}")
    marker = ".layers."
    if marker not in prefix or ".self_attn.attn" not in prefix:
        raise RuntimeError(f"R003e cannot parse attention prefix: {prefix!r}")
    return int(prefix.split(marker, 1)[1].split(".", 1)[0])


def _patch_lfm2(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[LFM2_MODULE](module)
    if not ENABLED:
        return
    attention_base = getattr(module, "Attention", None)
    if attention_base is None:
        raise RuntimeError("R003e could not find LFM2 Attention global")
    if getattr(attention_base, _LAYER_MARKER, False):
        return

    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    class PrefixRecentAttention(attention_base):
        _r003e_prefix_recent = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            layer_index = _layer_index(kwargs.get("prefix"))
            if layer_index not in WINDOW_MAP:
                raise RuntimeError(f"R003e unexpected attention layer {layer_index}")
            existing = kwargs.get("per_layer_sliding_window")
            if existing is not None and int(existing) != WINDOW_MAP[layer_index]:
                raise RuntimeError(
                    f"R003e conflicting window layer={layer_index} "
                    f"{existing} != {WINDOW_MAP[layer_index]}"
                )
            existing_backend = kwargs.get("attn_backend")
            if existing_backend is not None and existing_backend is not TritonAttentionBackend:
                raise RuntimeError("R003e refuses a non-Triton preselected backend")
            kwargs["per_layer_sliding_window"] = WINDOW_MAP[layer_index]
            kwargs["attn_backend"] = TritonAttentionBackend
            super().__init__(*args, **kwargs)
            if getattr(self.attn_backend, "get_name")() != "TRITON_ATTN":
                raise RuntimeError("R003e Triton backend was not installed")
            self.use_mm_prefix = PREFIX_ENABLED
            self._r003e_layer_index = layer_index
            self._r003e_window = WINDOW_MAP[layer_index]

    module.Attention = PrefixRecentAttention
    base_model = getattr(module, "Lfm2ForCausalLM", None)
    if base_model is None:
        raise RuntimeError("R003e could not find Lfm2ForCausalLM")
    if getattr(base_model, "_r003e_window_verified", False):
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
                marker = getattr(attention, "_r003e_layer_index", None)
                if marker != index:
                    raise RuntimeError(
                        f"R003e layer marker mismatch model={index} attention={marker}"
                    )
                observed[index] = (
                    int(getattr(attention, "_r003e_window", -1)),
                    str(getattr(attention.attn_backend, "get_name")()),
                )
            elif getattr(layer, "short_conv", None) is not None:
                shortconv.append(index)
        expected = {index: (WINDOW_MAP[index], "TRITON_ATTN") for index in FULL_LAYERS}
        if observed != expected or len(shortconv) != 10:
            raise RuntimeError(
                f"R003e topology verification failed expected={expected} "
                f"observed={observed} shortconv={shortconv}"
            )
        base_model._r003e_window_verified = True
        map_text = ",".join(f"{i}:{WINDOW_MAP[i]}" for i in sorted(WINDOW_MAP))
        print(
            "R003_WINDOW verified=1 "
            f"window={min(WINDOW_MAP.values())} tiered_map={map_text} "
            f"backend=TRITON_ATTN shortconv_layers={shortconv}",
            file=sys.stderr,
            flush=True,
        )

    base_model.__init__ = init_and_verify
    base_model._r003e_window_verified = False
    map_text = ",".join(f"{i}:{WINDOW_MAP[i]}" for i in sorted(WINDOW_MAP))
    print(
        "R003_WINDOW lfm2_patched=1 enabled=1 "
        f"window={min(WINDOW_MAP.values())} tiered_map={map_text} "
        f"backend=TRITON_ATTN prefix_tokens={PREFIX_TOKENS}",
        file=sys.stderr,
        flush=True,
    )


def _patch_sampler(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[SAMPLER_MODULE](module)


def _metadata_objects(attn_metadata: Any) -> list[Any]:
    if isinstance(attn_metadata, list):
        objects: list[Any] = []
        for item in attn_metadata:
            objects.extend(item.values())
        return objects
    if isinstance(attn_metadata, dict):
        return list(attn_metadata.values())
    return []


def _patch_runner(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[RUNNER_MODULE](module)
    # R003e reuses the R002y panel protocol.  Its profiler must be composed
    # here because this overlay owns the runner import hook and shadows the
    # standalone R002y sitecustomize module.
    if os.environ.get("R002Y_PROFILE_ARM"):
        from prefill_profiler import _patch_runner as patch_profiler_runner

        patch_profiler_runner(module)
    base = getattr(module, "GPUModelRunner", None)
    if base is None or getattr(base, _RUNNER_MARKER, False) or not ENABLED:
        return

    native_init = base.__init__
    native_set_ranges = base._set_mm_prefix_range_for_metadata

    @functools.wraps(native_init)
    def init_prefix_aware(self: Any, *args: Any, **kwargs: Any) -> None:
        native_init(self, *args, **kwargs)
        # The model has no multimodal documents, but the Triton metadata path
        # is reusable for a deterministic text prefix range.
        self.is_mm_prefix_lm = PREFIX_ENABLED

    @functools.wraps(native_set_ranges)
    def set_prefix_ranges(
        self: Any, attn_metadata: Any, _request_ranges: dict[int, list[tuple[int, int]]]
    ) -> None:
        global _ANNOUNCED_METADATA
        if not PREFIX_ENABLED:
            native_set_ranges(self, attn_metadata, _request_ranges)
            return
        objects = _metadata_objects(attn_metadata)
        max_requests = max(
            (int(getattr(obj, "seq_lens").shape[0]) for obj in objects if hasattr(obj, "seq_lens")),
            default=0,
        )
        if max_requests == 0:
            native_set_ranges(self, attn_metadata, _request_ranges)
            return
        ranges = prefix_ranges(max_requests, PREFIX_TOKENS)
        native_set_ranges(self, attn_metadata, ranges)
        if not _ANNOUNCED_METADATA:
            _ANNOUNCED_METADATA = True
            triton_objects = sum(
                1 for obj in objects if hasattr(obj, "mm_prefix_range_tensor")
            )
            print(
                "R003E_PREFIX_RANGE active=1 "
                f"prefix_tokens={PREFIX_TOKENS} requests={max_requests} "
                f"triton_metadata_objects={triton_objects}",
                file=sys.stderr,
                flush=True,
            )

    base.__init__ = init_prefix_aware
    base._set_mm_prefix_range_for_metadata = set_prefix_ranges
    setattr(base, _RUNNER_MARKER, True)
    print(
        f"R003E runner_patched=1 prefix_metadata={int(PREFIX_ENABLED)}",
        file=sys.stderr,
        flush=True,
    )


PATCHERS = {
    LFM2_MODULE: _patch_lfm2,
    SAMPLER_MODULE: _patch_sampler,
    RUNNER_MODULE: _patch_runner,
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
        "R003E hook_installed=1 "
        f"prefix_tokens={PREFIX_TOKENS} map={map_text} backend=TRITON_ATTN",
        file=sys.stderr,
        flush=True,
    )
