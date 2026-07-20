"""Decode-only prefix/recent cascade overlay for the native FlashInfer path.

This is intentionally conservative. It activates only when every request in a
decode batch has a page-aligned sequence length large enough for two disjoint
segments. Prefill, partial pages and mixed eligibility stay on native vLLM.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import itertools
import os
import sys
from types import ModuleType
from typing import Any

import fused_greedy_head
import torch


LFM2_MODULE = fused_greedy_head.LFM2_MODULE
SAMPLER_MODULE = fused_greedy_head.SAMPLER_MODULE
RUNNER_MODULE = fused_greedy_head.RUNNER_MODULE
FLASHINFER_MODULE = "vllm.v1.attention.backends.flashinfer"
FULL_LAYERS = frozenset({2, 5, 8, 10, 12, 14})
ENABLED = os.environ.get("R003H_ENABLED", "1") == "1"
WINDOW = int(os.environ.get("R003H_WINDOW", "4096"))
PREFIX = int(os.environ.get("R003H_PREFIX_TOKENS", "128"))
if WINDOW <= 0 or PREFIX < 0:
    raise ValueError("R003H_WINDOW must be positive and PREFIX non-negative")
if WINDOW % 16 or PREFIX % 16:
    raise ValueError("R003H prefix and window must be multiples of 16")
CASCADE_ENABLED = os.environ.get("R003H_CASCADE", "1") == "1"

_LAYER_MARKER = "_r003h_flashinfer_cascade_layer"
_BUILDER_MARKER = "_r003h_flashinfer_cascade_builder"
_RUNNER_MARKER = "_r003h_flashinfer_cascade_runner"
_ANNOUNCED = False
_ANNOUNCED_SKIP = False


def _layer_index(prefix: Any) -> int:
    if not isinstance(prefix, str):
        raise RuntimeError(f"R003h missing attention prefix: {prefix!r}")
    marker = ".layers."
    if marker not in prefix or ".self_attn.attn" not in prefix:
        raise RuntimeError(f"R003h cannot parse attention prefix: {prefix!r}")
    return int(prefix.split(marker, 1)[1].split(".", 1)[0])


def _patch_lfm2(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[LFM2_MODULE](module)
    if not ENABLED:
        return
    attention_base = getattr(module, "Attention", None)
    if attention_base is None or getattr(attention_base, _LAYER_MARKER, False):
        return

    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    class FlashInferCascadeAttention(attention_base):
        _r003h_flashinfer_cascade_layer = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            layer_index = _layer_index(kwargs.get("prefix"))
            if layer_index not in FULL_LAYERS:
                raise RuntimeError(f"R003h unexpected full-attention layer {layer_index}")
            existing = kwargs.get("per_layer_sliding_window")
            if existing is not None and int(existing) != WINDOW:
                raise RuntimeError(
                    f"R003h conflicting window layer={layer_index} {existing} != {WINDOW}"
                )
            existing_backend = kwargs.get("attn_backend")
            if existing_backend is not None and existing_backend is not FlashInferBackend:
                raise RuntimeError(
                    f"R003h refuses non-FlashInfer backend: {existing_backend}"
                )
            kwargs["per_layer_sliding_window"] = WINDOW
            kwargs["attn_backend"] = FlashInferBackend
            super().__init__(*args, **kwargs)
            if self.attn_backend.get_name() != "FLASHINFER":
                raise RuntimeError("R003h FlashInfer backend was not installed")
            self._r003h_layer_index = layer_index
            self._r003h_window = WINDOW

    module.Attention = FlashInferCascadeAttention
    base_model = getattr(module, "Lfm2ForCausalLM", None)
    if base_model is None:
        raise RuntimeError("R003h could not find Lfm2ForCausalLM")
    if getattr(base_model, "_r003h_topology_verified", False):
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
                marker = getattr(attention, "_r003h_layer_index", None)
                if marker != index:
                    raise RuntimeError(
                        f"R003h layer marker mismatch model={index} attention={marker}"
                    )
                observed[index] = (
                    int(getattr(attention, "_r003h_window", -1)),
                    str(attention.attn_backend.get_name()),
                )
            elif getattr(layer, "short_conv", None) is not None:
                shortconv.append(index)
        expected = {index: (WINDOW, "FLASHINFER") for index in FULL_LAYERS}
        if observed != expected or len(shortconv) != 10:
            raise RuntimeError(
                f"R003h topology verification failed expected={expected} "
                f"observed={observed} shortconv={shortconv}"
            )
        base_model._r003h_topology_verified = True
        print(
            "R003H verified=1 attention_layers=6 shortconv_layers=10 "
            f"window={WINDOW} backend=FLASHINFER prefix={PREFIX} "
            f"cascade={int(CASCADE_ENABLED)}",
            file=sys.stderr,
            flush=True,
        )

    base_model.__init__ = init_and_verify
    base_model._r003h_topology_verified = False


def _device_tensor(value: Any, *, device: Any, dtype: Any) -> Any:
    if value.device == device and value.dtype == dtype:
        return value
    return value.to(device=device, dtype=dtype)


def _build_decode_cascade(builder: Any, common: Any, metadata: Any) -> bool:
    """Install two page-aligned levels for an all-single-token batch."""
    global _ANNOUNCED, _ANNOUNCED_SKIP
    if not CASCADE_ENABLED or metadata.use_cascade:
        return False
    query_start_loc = getattr(common, "query_start_loc", None)
    if query_start_loc is None or query_start_loc.numel() < 2:
        return False
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    if not bool(torch_all_equal(query_lens, 1)):
        return False
    num_reqs = int(query_lens.numel())
    seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens = getattr(common, "seq_lens", None)
        if seq_lens is None:
            return False
        seq_lens_cpu = seq_lens.detach().cpu()
    seq_lens_list = [int(item) for item in seq_lens_cpu.tolist()]
    page_size = int(builder.page_size)
    if PREFIX % page_size or WINDOW % page_size:
        return False
    prefix_pages = PREFIX // page_size
    base_recent_pages = WINDOW // page_size
    if prefix_pages <= 0 or base_recent_pages <= 0:
        return False
    if any(
        seq_len < PREFIX + WINDOW + page_size
        for seq_len in seq_lens_list
    ):
        if not _ANNOUNCED_SKIP:
            _ANNOUNCED_SKIP = True
            print(
                "R003H cascade_skip=1 reason=non_aligned_or_short_decode",
                file=sys.stderr,
                flush=True,
            )
        return False

    block_table = common.block_table_tensor
    device = block_table.device
    prefix_indices = block_table[:, :prefix_pages].reshape(-1).contiguous()
    recent_rows = []
    recent_page_counts: list[int] = []
    recent_last_lengths: list[int] = []
    for row, seq_len in zip(block_table, seq_lens_list, strict=True):
        num_pages = seq_len // page_size
        last_len = seq_len % page_size or page_size
        recent_pages = base_recent_pages + int(last_len != page_size)
        recent_page_counts.append(recent_pages)
        recent_last_lengths.append(last_len)
        recent_rows.append(row[num_pages - recent_pages : num_pages])
    recent_indices = torch.cat(recent_rows).contiguous()
    prefix_indptr = torch.arange(
        0,
        (num_reqs + 1) * prefix_pages,
        prefix_pages,
        dtype=torch.int32,
        device=device,
    )
    recent_indptr = torch.tensor(
        [0]
        + list(
            itertools.accumulate(recent_page_counts)
        ),
        dtype=torch.int32,
        device=device,
    )
    recent_last_page_len = torch.tensor(
        recent_last_lengths, dtype=torch.int32, device=device
    )
    prefix_last_page_len = torch.full(
        (num_reqs,), page_size, dtype=torch.int32, device=device
    )
    qo_indptr = _device_tensor(query_start_loc, device=device, dtype=torch.int32)
    wrapper = builder._get_cascade_wrapper()
    wrapper.plan(
        qo_indptr_arr=[qo_indptr, qo_indptr],
        paged_kv_indptr_arr=[prefix_indptr, recent_indptr],
        paged_kv_indices_arr=[prefix_indices, recent_indices],
        paged_kv_last_page_len=[prefix_last_page_len, recent_last_page_len],
        num_qo_heads=builder.num_qo_heads,
        num_kv_heads=builder.num_kv_heads,
        head_dim=builder.head_dim,
        page_size=page_size,
        causal=True,
        sm_scale=builder.sm_scale,
        window_left=-1,
        logits_soft_cap=builder.logits_soft_cap,
        q_data_type=builder.q_data_type,
        kv_data_type=builder.kv_cache_dtype,
    )
    metadata.use_cascade = True
    metadata.cascade_wrapper = wrapper
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(
            "R003H_CASCADE active=1 mode=decode_only page_safe=1 "
            f"batch={num_reqs} prefix={PREFIX} recent={WINDOW} "
            f"backend=FLASHINFER kv_dtype={builder.kv_cache_dtype} "
            f"recent_pages={sorted(set(recent_page_counts))}",
            file=sys.stderr,
            flush=True,
        )
    return True


def torch_all_equal(values: Any, expected: int) -> bool:
    return bool((values == expected).all().item())


def _patch_flashinfer(module: ModuleType) -> None:
    builder_cls = getattr(module, "FlashInferMetadataBuilder", None)
    if builder_cls is None or getattr(builder_cls, _BUILDER_MARKER, False):
        return
    native_build = builder_cls.build

    @functools.wraps(native_build)
    def build_with_cascade(
        self: Any, common_prefix_len: int, common_attn_metadata: Any, fast_build: bool = False
    ) -> Any:
        metadata = native_build(
            self,
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )
        if ENABLED and metadata is not None:
            try:
                _build_decode_cascade(self, common_attn_metadata, metadata)
            except Exception as exc:
                print(
                    f"R003H cascade_skip=1 reason=exception type={type(exc).__name__} "
                    f"message={exc}",
                    file=sys.stderr,
                    flush=True,
                )
        return metadata

    builder_cls.build = build_with_cascade
    setattr(builder_cls, _BUILDER_MARKER, True)
    print(
        "R003H builder_patched=1 backend=FLASHINFER decode_only=1",
        file=sys.stderr,
        flush=True,
    )


def _patch_sampler(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[SAMPLER_MODULE](module)


def _patch_runner(module: ModuleType) -> None:
    fused_greedy_head.PATCHERS[RUNNER_MODULE](module)
    if os.environ.get("R002Y_PROFILE_ARM"):
        from prefill_profiler import _patch_runner as patch_profiler_runner

        patch_profiler_runner(module)
    base = getattr(module, "GPUModelRunner", None)
    if base is None or getattr(base, _RUNNER_MARKER, False):
        return
    setattr(base, _RUNNER_MARKER, True)
    print("R003H runner_patched=1", file=sys.stderr, flush=True)


PATCHERS = {
    LFM2_MODULE: _patch_lfm2,
    SAMPLER_MODULE: _patch_sampler,
    RUNNER_MODULE: _patch_runner,
    FLASHINFER_MODULE: _patch_flashinfer,
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
        "R003H hook_installed=1 enabled=1 backend=FLASHINFER "
        f"window={WINDOW} prefix={PREFIX} cascade={int(CASCADE_ENABLED)}",
        file=sys.stderr,
        flush=True,
    )
