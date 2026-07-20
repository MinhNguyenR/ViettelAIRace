"""Opt-in probe for vLLM's existing exact shared-prefix cascade path.

The overlay does not construct a new KV layout. It only replaces the
FlashInfer builder's disabled admission predicate with the upstream heuristic
used by the FlashAttention backend and records the resulting decisions. The
native FlashInfer ``build`` method remains responsible for the two-level
metadata and online-softmax merge.
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
import numpy as np


LFM2_MODULE = fused_greedy_head.LFM2_MODULE
SAMPLER_MODULE = fused_greedy_head.SAMPLER_MODULE
RUNNER_MODULE = fused_greedy_head.RUNNER_MODULE
FLASHINFER_MODULE = "vllm.v1.attention.backends.flashinfer"
FLASH_ATTN_MODULE = "vllm.v1.attention.backends.flash_attn"
ENABLED = os.environ.get("R003I_ENABLED", "1") == "1"
FORCE = os.environ.get("R003I_FORCE", "0") == "1"
ALLOW_FP8 = os.environ.get("R003I_ALLOW_FP8", "0") == "1"
FORCE_FLASHINFER = os.environ.get("R003I_FORCE_FLASHINFER", "1") == "1"
MIN_PREFIX = int(os.environ.get("R003I_MIN_PREFIX", "256"))
FULL_LAYERS = frozenset({2, 5, 8, 10, 12, 14})
_LAYER_MARKER = "_r003i_flashinfer_layer"
_BUILDER_MARKER = "_r003i_shared_prefix_builder"
_FA_BUILDER_MARKER = "_r003i_shared_prefix_fa_builder"
_RUNNER_MARKER = "_r003i_shared_prefix_runner"
_DECISION_COUNT = 0
_BUILD_COUNT = 0
_CASCADE_BUILD_COUNT = 0
_NONZERO_COUNT = 0
_PREFIX_COUNT = 0
_FA_DECISION_COUNT = 0
_FA_CASCADE_BUILD_COUNT = 0


def _patch_lfm2(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[LFM2_MODULE](module)
    if not FORCE_FLASHINFER:
        return
    attention_base = getattr(module, "Attention", None)
    if attention_base is None or getattr(attention_base, _LAYER_MARKER, False):
        return

    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    def layer_index(prefix: Any) -> int:
        if not isinstance(prefix, str) or ".layers." not in prefix:
            raise RuntimeError(f"R003i cannot parse attention prefix: {prefix!r}")
        return int(prefix.split(".layers.", 1)[1].split(".", 1)[0])

    class FlashInferSharedPrefixAttention(attention_base):
        _r003i_flashinfer_layer = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            index = layer_index(kwargs.get("prefix"))
            if index not in FULL_LAYERS:
                raise RuntimeError(f"R003i unexpected attention layer {index}")
            existing_window = kwargs.get("per_layer_sliding_window")
            if existing_window not in (None, -1):
                raise RuntimeError(
                    f"R003i refuses a sliding-window layer: {index} {existing_window}"
                )
            kwargs["attn_backend"] = FlashInferBackend
            super().__init__(*args, **kwargs)
            if self.attn_backend.get_name() != "FLASHINFER":
                raise RuntimeError("R003i FlashInfer backend was not installed")
            self._r003i_layer_index = index

    module.Attention = FlashInferSharedPrefixAttention
    base_model = getattr(module, "Lfm2ForCausalLM", None)
    if base_model is None:
        raise RuntimeError("R003i could not find Lfm2ForCausalLM")
    if getattr(base_model, "_r003i_topology_verified", False):
        return
    native_init = base_model.__init__

    @functools.wraps(native_init)
    def init_and_verify(self: Any, *args: Any, **kwargs: Any) -> None:
        native_init(self, *args, **kwargs)
        model = getattr(self, "model", None)
        layers = list(getattr(model, "layers", []))
        observed: dict[int, str] = {}
        shortconv: list[int] = []
        for index, layer in enumerate(layers):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None:
                attention = getattr(self_attn, "attn", None)
                marker = getattr(attention, "_r003i_layer_index", None)
                if marker != index:
                    raise RuntimeError(
                        f"R003i layer marker mismatch model={index} attention={marker}"
                    )
                observed[index] = str(attention.attn_backend.get_name())
            elif getattr(layer, "short_conv", None) is not None:
                shortconv.append(index)
        expected = {index: "FLASHINFER" for index in FULL_LAYERS}
        if observed != expected or len(shortconv) != 10:
            raise RuntimeError(
                f"R003i topology verification failed expected={expected} "
                f"observed={observed} shortconv={shortconv}"
            )
        base_model._r003i_topology_verified = True
        print(
            "R003I verified=1 attention_layers=6 shortconv_layers=10 "
            "backend=FLASHINFER window=none",
            file=sys.stderr,
            flush=True,
        )

    base_model.__init__ = init_and_verify
    base_model._r003i_topology_verified = False


def _patch_sampler(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[SAMPLER_MODULE](module)


def _patch_runner(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[RUNNER_MODULE](module)
    if os.environ.get("R002Y_PROFILE_ARM"):
        from prefill_profiler import _patch_runner as patch_profiler_runner

        patch_profiler_runner(module)
    base = getattr(module, "GPUModelRunner", None)
    if base is not None and not getattr(base, _RUNNER_MARKER, False):
        native_prefix = getattr(base, "_compute_cascade_attn_prefix_len", None)
        if native_prefix is not None:
            @functools.wraps(native_prefix)
            def observe_prefix(
                self: Any,
                num_scheduled_tokens: Any,
                num_computed_tokens: Any,
                num_common_prefix_blocks: int,
                kv_cache_spec: Any,
                attn_metadata_builder: Any,
            ) -> int:
                global _PREFIX_COUNT
                result = native_prefix(
                    self,
                    num_scheduled_tokens,
                    num_computed_tokens,
                    num_common_prefix_blocks,
                    kv_cache_spec,
                    attn_metadata_builder,
                )
                _PREFIX_COUNT += 1
                if _PREFIX_COUNT <= 100 and (
                    num_common_prefix_blocks > 0 or _PREFIX_COUNT <= 8
                ):
                    print(
                        "R003I_PREFIX "
                        f"blocks={num_common_prefix_blocks} result={result} "
                        f"scheduled={','.join(str(int(x)) for x in num_scheduled_tokens[:16])} "
                        f"computed_min={int(np.min(num_computed_tokens))}",
                        file=sys.stderr,
                        flush=True,
                    )
                return result

            base._compute_cascade_attn_prefix_len = observe_prefix
        setattr(base, _RUNNER_MARKER, True)
        print("R003I runner_patched=1", file=sys.stderr, flush=True)


def _patch_flashinfer(module: ModuleType) -> None:
    builder_cls = getattr(module, "FlashInferMetadataBuilder", None)
    if builder_cls is None or getattr(builder_cls, _BUILDER_MARKER, False):
        return

    from vllm.v1.attention.backends.flash_attn import (
        use_cascade_attention as upstream_heuristic,
    )

    native_use = builder_cls.use_cascade_attention
    native_build = builder_cls.build

    @functools.wraps(native_use)
    def use_shared_prefix(
        self: Any,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
        dcp_world_size: int,
    ) -> bool:
        global _DECISION_COUNT, _NONZERO_COUNT
        if not ENABLED:
            return native_use(
                self,
                common_prefix_len,
                query_lens,
                num_query_heads,
                num_kv_heads,
                use_alibi,
                use_sliding_window,
                use_local_attention,
                num_sms,
                dcp_world_size,
            )

        # The shipped source explicitly rejects a KV dtype different from the
        # model dtype. Keep that boundary visible unless the experiment opts
        # into an FP8 compatibility trial.
        dtype = getattr(getattr(self, "kv_cache_spec", None), "dtype", None)
        model_dtype = getattr(getattr(self, "vllm_config", None), "model_config", None)
        model_dtype = getattr(model_dtype, "dtype", None)
        dtype_ok = ALLOW_FP8 or dtype == model_dtype
        if not dtype_ok:
            decision = False
            reason = "kv_dtype_mismatch"
        elif FORCE and common_prefix_len >= MIN_PREFIX:
            decision = (
                not use_alibi
                and not use_sliding_window
                and not use_local_attention
                and dcp_world_size == 1
                and len(query_lens) >= 8
            )
            reason = "forced_threshold"
        else:
            decision = upstream_heuristic(
                common_prefix_len=common_prefix_len,
                query_lens=query_lens,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                use_alibi=use_alibi,
                use_sliding_window=use_sliding_window,
                use_local_attention=use_local_attention,
                num_sms=num_sms,
                dcp_world_size=dcp_world_size,
            )
            reason = "upstream_heuristic"

        _DECISION_COUNT += 1
        if decision:
            _NONZERO_COUNT += 1
        if _DECISION_COUNT <= 80 and (common_prefix_len > 0 or _DECISION_COUNT <= 8):
            print(
                "R003I_DECISION "
                f"prefix={common_prefix_len} batch={len(query_lens)} "
                f"q_heads={num_query_heads} kv_heads={num_kv_heads} "
                f"num_sms={num_sms} "
                f"q_lens={','.join(str(int(x)) for x in query_lens[:16])} "
                f"dtype_ok={int(dtype_ok)} decision={int(decision)} "
                f"reason={reason} kv_dtype={dtype!s} model_dtype={model_dtype!s}",
                file=sys.stderr,
                flush=True,
            )
        return decision

    @functools.wraps(native_build)
    def build_with_observer(
        self: Any,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
    ) -> Any:
        global _BUILD_COUNT, _CASCADE_BUILD_COUNT
        metadata = native_build(
            self,
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )
        _BUILD_COUNT += 1
        if common_prefix_len > 0:
            _CASCADE_BUILD_COUNT += 1
        if _CASCADE_BUILD_COUNT <= 80 and common_prefix_len > 0:
            query_start = getattr(common_attn_metadata, "query_start_loc_cpu", None)
            batch = int(query_start.numel() - 1) if query_start is not None else -1
            print(
                "R003I_BUILD "
                f"cascade_build={_CASCADE_BUILD_COUNT} total_build={_BUILD_COUNT} "
                f"prefix={common_prefix_len} batch={batch} "
                f"use_cascade={int(getattr(metadata, 'use_cascade', False))} "
                f"kv_dtype={getattr(self, 'kv_cache_dtype', None)!s}",
                file=sys.stderr,
                flush=True,
            )
        return metadata

    builder_cls.use_cascade_attention = use_shared_prefix
    builder_cls.build = build_with_observer
    setattr(builder_cls, _BUILDER_MARKER, True)
    print(
        "R003I builder_patched=1 backend=FLASHINFER exact_shared_prefix=1 "
        f"force={int(FORCE)} allow_fp8={int(ALLOW_FP8)} "
        f"force_backend={int(FORCE_FLASHINFER)} min_prefix={MIN_PREFIX}",
        file=sys.stderr,
        flush=True,
    )


def _patch_flash_attn(module: ModuleType) -> None:
    builder_cls = getattr(module, "FlashAttentionMetadataBuilder", None)
    if builder_cls is None or getattr(builder_cls, _FA_BUILDER_MARKER, False):
        return

    native_use = builder_cls.use_cascade_attention
    native_build = builder_cls.build

    @functools.wraps(native_use)
    def observe_use(
        self: Any,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
        dcp_world_size: int,
    ) -> bool:
        global _FA_DECISION_COUNT
        decision = native_use(
            self,
            common_prefix_len,
            query_lens,
            num_query_heads,
            num_kv_heads,
            use_alibi,
            use_sliding_window,
            use_local_attention,
            num_sms,
            dcp_world_size,
        )
        _FA_DECISION_COUNT += 1
        if _FA_DECISION_COUNT <= 80 and (
            common_prefix_len > 0 or _FA_DECISION_COUNT <= 8
        ):
            print(
                "R003I_FA_DECISION "
                f"prefix={common_prefix_len} batch={len(query_lens)} "
                f"q_heads={num_query_heads} kv_heads={num_kv_heads} "
                f"num_sms={num_sms} "
                f"q_lens={','.join(str(int(x)) for x in query_lens[:16])} "
                f"decision={int(decision)}",
                file=sys.stderr,
                flush=True,
            )
        return decision

    @functools.wraps(native_build)
    def observe_build(
        self: Any,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
    ) -> Any:
        global _FA_CASCADE_BUILD_COUNT
        metadata = native_build(
            self,
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )
        if common_prefix_len > 0:
            _FA_CASCADE_BUILD_COUNT += 1
        if _FA_CASCADE_BUILD_COUNT <= 80 and common_prefix_len > 0:
            print(
                "R003I_FA_BUILD "
                f"cascade_build={_FA_CASCADE_BUILD_COUNT} "
                f"prefix={common_prefix_len} "
                f"batch={int(common_attn_metadata.num_reqs)} "
                f"use_cascade={int(getattr(metadata, 'use_cascade', False))}",
                file=sys.stderr,
                flush=True,
            )
        return metadata

    builder_cls.use_cascade_attention = observe_use
    builder_cls.build = observe_build
    setattr(builder_cls, _FA_BUILDER_MARKER, True)
    print(
        "R003I fa_builder_patched=1 backend=FLASH_ATTN native_heuristic=1",
        file=sys.stderr,
        flush=True,
    )


PATCHERS = {
    LFM2_MODULE: _patch_lfm2,
    SAMPLER_MODULE: _patch_sampler,
    RUNNER_MODULE: _patch_runner,
    FLASHINFER_MODULE: _patch_flashinfer,
    FLASH_ATTN_MODULE: _patch_flash_attn,
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
    print(
        "R003I hook_installed=1 "
        f"enabled={int(ENABLED)} "
        f"backend={'FLASHINFER' if FORCE_FLASHINFER else 'FLASH_ATTN'} "
        f"force={int(FORCE)} allow_fp8={int(ALLOW_FP8)} "
        f"force_backend={int(FORCE_FLASHINFER)}",
        file=sys.stderr,
        flush=True,
    )
