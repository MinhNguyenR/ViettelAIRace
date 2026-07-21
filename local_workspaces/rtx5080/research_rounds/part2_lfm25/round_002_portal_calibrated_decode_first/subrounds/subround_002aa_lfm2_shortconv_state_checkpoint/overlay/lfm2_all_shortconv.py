"""Complete LFM2 ShortConv wiring for vLLM's existing all-mode state path."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType
from typing import Any, Callable

import torch


LFM2_MODULE = "vllm.model_executor.models.lfm2"
SHORT_CONV_MODULE = "vllm.model_executor.layers.mamba.short_conv"
MODEL_CONFIG_MODULE = "vllm.config.model"
MAMBA_METADATA_MODULE = "vllm.v1.attention.backends.mamba_attn"
MAMBA_MANAGER_MODULE = "vllm.v1.core.single_type_kv_cache_manager"
SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"
SAMPLER_MODULE = "vllm.v1.sample.sampler"
RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
_HOOK_INSTALLED = False
_SPARSE_RUNTIME_ENABLED = os.environ.get("R002AA_SPARSE_DUAL") == "1"
_SPARSE_MODE_FILE = os.environ.get("R002AA_MODE_FILE", "")


def _sparse_enabled() -> bool:
    return _SPARSE_RUNTIME_ENABLED


def _refresh_sparse_mode() -> None:
    global _SPARSE_RUNTIME_ENABLED
    if not _SPARSE_MODE_FILE:
        return
    try:
        with open(_SPARSE_MODE_FILE, encoding="utf-8") as handle:
            value = handle.read().strip()
    except FileNotFoundError:
        return
    if value not in {"0", "1"}:
        raise RuntimeError(f"invalid R002aa sparse mode file value: {value!r}")
    _SPARSE_RUNTIME_ENABLED = value == "1"


def _patch_model_config(module: ModuleType) -> None:
    base = getattr(module, "ModelConfig", None)
    if base is None or getattr(base, "_r002aa_all_mode_patched", False):
        return
    native_property = base.supports_mamba_prefix_caching
    native_getter = native_property.fget
    if native_getter is None:
        raise RuntimeError("R002aa expected a readable capability property")

    @property
    @functools.wraps(native_getter)
    def supports_mamba_prefix_caching(self: Any) -> bool:
        if self.architecture == "Lfm2ForCausalLM":
            return True
        return bool(native_getter(self))

    base.supports_mamba_prefix_caching = supports_mamba_prefix_caching
    base._r002aa_all_mode_patched = True
    print("R002AA_ALL_SHORTCONV config_patched=1", file=sys.stderr, flush=True)


def _patch_scheduler(module: ModuleType) -> None:
    base = getattr(module, "Scheduler", None)
    if base is None or getattr(base, "_r002aa_sparse_dual_patched", False):
        return
    native_split = base._mamba_block_aligned_split
    native_schedule = base.schedule

    @functools.wraps(native_schedule)
    def schedule_with_runtime_mode(self: Any, *args: Any, **kwargs: Any) -> Any:
        _refresh_sparse_mode()
        return native_schedule(self, *args, **kwargs)

    @functools.wraps(native_split)
    def split_with_sparse_final_step(
        self: Any,
        request: Any,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        if not _sparse_enabled():
            return native_split(
                self,
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        block_size = self.cache_config.block_size
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        num_computed_after = num_computed_tokens + num_new_tokens
        completes_request = num_computed_after >= request.num_tokens
        crosses_final_boundary = (
            num_computed_tokens < last_cache_position < num_computed_after
        )
        if (
            num_computed_tokens > 0
            and completes_request
            and crosses_final_boundary
        ):
            return num_new_tokens
        return native_split(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )

    base._mamba_block_aligned_split = split_with_sparse_final_step
    base.schedule = schedule_with_runtime_mode
    base._r002aa_sparse_dual_patched = True
    print("R002AA_SPARSE_DUAL scheduler_patched=1", file=sys.stderr, flush=True)


def _patch_mamba_manager(module: ModuleType) -> None:
    base = getattr(module, "MambaManager", None)
    if base is None or getattr(base, "_r002aa_sparse_dual_patched", False):
        return
    native_count = base.get_num_blocks_to_allocate
    native_allocate = base.allocate_new_blocks

    @functools.wraps(native_count)
    def count_with_boundary(
        self: Any,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Any,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        count = native_count(
            self,
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )
        if (
            not _sparse_enabled()
            or self.mamba_cache_mode != "align"
            or request_id in self._allocated_block_reqs
        ):
            return count
        block_size = self.block_size
        final_boundary = (
            num_tokens_main_model - num_tokens_main_model % block_size
        )
        if 0 < total_computed_tokens < final_boundary < num_tokens_main_model:
            return count + 1
        return count

    @functools.wraps(native_allocate)
    def allocate_with_boundary(
        self: Any,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[Any]:
        if not _sparse_enabled() or self.mamba_cache_mode != "align":
            return native_allocate(
                self, request_id, num_tokens, num_tokens_main_model
            )
        was_allocated = request_id in self._allocated_block_reqs
        new_blocks = native_allocate(
            self, request_id, num_tokens, num_tokens_main_model
        )
        if was_allocated:
            return new_blocks

        block_size = self.block_size
        if num_tokens_main_model % block_size == 0:
            return new_blocks
        boundary_idx = num_tokens_main_model // block_size - 1
        final_idx = (num_tokens_main_model + block_size - 1) // block_size - 1
        if boundary_idx < 0 or boundary_idx == final_idx:
            return new_blocks
        request_blocks = self.req_to_blocks[request_id]
        if final_idx >= len(request_blocks):
            raise RuntimeError("R002aa final state block was not allocated")
        if not request_blocks[boundary_idx].is_null:
            return new_blocks

        boundary_block = self.block_pool.get_new_blocks(1)[0]
        request_blocks[boundary_idx] = boundary_block
        if os.environ.get("R002AA_DEBUG_BOUNDARY") == "1":
            print(
                "R002AA_SPARSE_DUAL boundary_allocated=1 "
                f"request={request_id} boundary_idx={boundary_idx} "
                f"final_idx={final_idx}",
                file=sys.stderr,
                flush=True,
            )
        return [boundary_block, *new_blocks]

    base.get_num_blocks_to_allocate = count_with_boundary
    base.allocate_new_blocks = allocate_with_boundary
    base._r002aa_sparse_dual_patched = True
    print("R002AA_SPARSE_DUAL manager_patched=1", file=sys.stderr, flush=True)


def _patch_mamba_metadata(module: ModuleType) -> None:
    base = getattr(module, "BaseMambaAttentionMetadataBuilder", None)
    if base is None or getattr(base, "_r002aa_sparse_dual_patched", False):
        return
    native_compute = base._compute_common_metadata

    @functools.wraps(native_compute)
    def compute_with_sparse_table(
        self: Any,
        common_attn_metadata: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        metadata = native_compute(self, common_attn_metadata, *args, **kwargs)
        if (
            not _sparse_enabled()
            or self.vllm_config.cache_config.mamba_cache_mode != "align"
            or metadata.num_prefills == 0
        ):
            return metadata

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if seq_lens_cpu is None:
            return metadata
        query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
        computed_cpu = seq_lens_cpu - query_lens_cpu
        prefill_start = common_attn_metadata.num_reqs - metadata.num_prefills
        seq_lens_p_cpu = seq_lens_cpu[prefill_start:]
        computed_p_cpu = computed_cpu[prefill_start:]
        final_boundaries_cpu = (
            seq_lens_p_cpu - seq_lens_p_cpu % self.kv_cache_spec.block_size
        )
        crosses_boundary_cpu = (
            (computed_p_cpu > 0)
            & (computed_p_cpu < final_boundaries_cpu)
            & (final_boundaries_cpu < seq_lens_p_cpu)
        )
        if not torch.any(crosses_boundary_cpu).item():
            return metadata

        native_state_indices = metadata.state_indices_tensor_p
        if native_state_indices is None:
            raise RuntimeError("R002aa missing native prefill state indices")
        native_state_indices = native_state_indices.reshape(metadata.num_prefills)

        num_reqs = metadata.num_reqs
        num_prefills = metadata.num_prefills
        block_size = self.kv_cache_spec.block_size
        num_computed = common_attn_metadata.compute_num_computed_tokens()
        computed_p = num_computed[num_reqs - num_prefills : num_reqs]
        seq_lens_p = common_attn_metadata.seq_lens[
            num_reqs - num_prefills : num_reqs
        ]
        boundary_logical = torch.clamp(seq_lens_p // block_size - 1, min=0)
        table_p = common_attn_metadata.block_table_tensor[
            num_reqs - num_prefills : num_reqs
        ]
        boundary_ids = table_p.gather(1, boundary_logical.unsqueeze(1)).squeeze(1)
        compact_state_indices = torch.stack(
            [native_state_indices, boundary_ids, native_state_indices], dim=1
        )
        crosses_boundary = (
            (computed_p > 0)
            & (computed_p < (seq_lens_p - seq_lens_p % block_size))
            & (seq_lens_p % block_size != 0)
        )
        first_indices = torch.where(
            crosses_boundary,
            torch.ones_like(computed_p),
            torch.full_like(computed_p, 2),
        )

        metadata.r002aa_sparse_dual = True
        metadata.r002aa_state_indices_p = compact_state_indices
        metadata.r002aa_initial_indices_p = torch.zeros_like(computed_p)
        metadata.r002aa_first_indices_p = first_indices
        metadata.r002aa_last_indices_p = torch.full_like(computed_p, 2)
        metadata.r002aa_num_computed_tokens_p = computed_p
        return metadata

    base._compute_common_metadata = compute_with_sparse_table
    base._r002aa_sparse_dual_patched = True
    print("R002AA_SPARSE_DUAL metadata_patched=1", file=sys.stderr, flush=True)


def _patch_lfm2(module: ModuleType) -> None:
    from fused_greedy_head import _patch_lfm2 as patch_r002k_lfm2

    patch_r002k_lfm2(module)
    base = getattr(module, "Lfm2ForCausalLM", None)
    if base is None or getattr(base, "_r002aa_all_mode_patched", False):
        return

    native_init = base.__init__

    @functools.wraps(native_init)
    def init_with_all_mode(self: Any, *args: Any, **kwargs: Any) -> None:
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None:
            raise RuntimeError("R002aa expected keyword-only vllm_config")
        cache_config = vllm_config.cache_config
        requested_mode = cache_config.mamba_cache_mode
        if requested_mode != "all":
            native_init(self, *args, **kwargs)
            return

        # LFM2 rejects all-mode before constructing ShortConv layers even though
        # the generic metadata and causal-conv kernels already implement it.
        # Temporarily pass that constructor guard, then restore the shared config
        # object before cache specs and runtime metadata are consumed.
        cache_config.mamba_cache_mode = "align"
        try:
            native_init(self, *args, **kwargs)
        finally:
            cache_config.mamba_cache_mode = requested_mode
        print(
            "R002AA_ALL_SHORTCONV lfm2_all_enabled=1",
            file=sys.stderr,
            flush=True,
        )

    base.__init__ = init_with_all_mode
    base._r002aa_all_mode_patched = True
    print("R002AA_ALL_SHORTCONV lfm2_patched=1", file=sys.stderr, flush=True)


def _patch_short_conv(module: ModuleType) -> None:
    base = getattr(module, "ShortConv", None)
    if base is None or getattr(base, "_r002aa_all_mode_patched", False):
        return

    native_forward_cuda = base.forward_cuda

    @functools.wraps(native_forward_cuda)
    def forward_cuda_all_mode(
        self: Any,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> Any:
        cache_config = self.cache_config
        if cache_config is None:
            return native_forward_cuda(self, hidden_states, output)
        if cache_config.mamba_cache_mode != "all" and not _sparse_enabled():
            return native_forward_cuda(self, hidden_states, output)

        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn,
            causal_conv1d_update,
        )
        from vllm.v1.attention.backends.short_conv_attn import (
            ShortConvAttentionMetadata,
        )

        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            return native_forward_cuda(self, hidden_states, output)
        if not isinstance(attn_metadata_raw, dict):
            raise RuntimeError("R002aa expected dict attention metadata")
        attn_metadata = attn_metadata_raw[self.prefix]
        if not isinstance(attn_metadata, ShortConvAttentionMetadata):
            raise RuntimeError("R002aa received unexpected ShortConv metadata")
        is_all_mode = cache_config.mamba_cache_mode == "all"
        is_sparse_mode = bool(
            getattr(attn_metadata, "r002aa_sparse_dual", False)
        )
        if not is_all_mode and not is_sparse_mode:
            return native_forward_cuda(self, hidden_states, output)

        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        state_indices_tensor_p = attn_metadata.state_indices_tensor_p
        state_indices_tensor_d = attn_metadata.state_indices_tensor_d
        has_initial_states_p = attn_metadata.has_initial_states_p
        query_start_loc_p = attn_metadata.query_start_loc_p

        projected, _ = self.in_proj(hidden_states)
        gate_b, gate_c, projected_x = projected.chunk(3, dim=-1)
        conv_weights = self.conv.weight.view(
            self.conv.weight.size(0), self.conv.weight.size(2)
        )

        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decodes
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        if num_decodes != num_decode_tokens:
            raise RuntimeError("R002aa all-mode proof does not enable spec decode")
        num_actual_tokens = num_decode_tokens + num_prefill_tokens

        b_decode, b_prefill = torch.split(
            gate_b[:num_actual_tokens],
            [num_decode_tokens, num_prefill_tokens],
            dim=0,
        )
        c_decode, c_prefill = torch.split(
            gate_c[:num_actual_tokens],
            [num_decode_tokens, num_prefill_tokens],
            dim=0,
        )
        x_decode, x_prefill = torch.split(
            projected_x[:num_actual_tokens],
            [num_decode_tokens, num_prefill_tokens],
            dim=0,
        )

        if is_all_mode:
            last_computed = attn_metadata.block_idx_last_computed_token
            last_scheduled = attn_metadata.block_idx_last_scheduled_token
            first_scheduled_p = attn_metadata.block_idx_first_scheduled_token_p
            computed_tokens_p = attn_metadata.num_computed_tokens_p
            if last_computed is None or last_scheduled is None:
                raise RuntimeError("R002aa all-mode block metadata is missing")
            last_computed_d, last_computed_p = torch.split(
                last_computed, [num_decodes, num_prefills], dim=0
            )
            last_scheduled_d, last_scheduled_p = torch.split(
                last_scheduled, [num_decodes, num_prefills], dim=0
            )
        else:
            state_indices_tensor_p = attn_metadata.r002aa_state_indices_p
            first_scheduled_p = attn_metadata.r002aa_first_indices_p
            last_scheduled_p = attn_metadata.r002aa_last_indices_p
            last_computed_p = attn_metadata.r002aa_initial_indices_p
            computed_tokens_p = attn_metadata.r002aa_num_computed_tokens_p
            last_scheduled_d = None
            last_computed_d = None

        conv_outputs: list[torch.Tensor] = []
        if num_prefill_tokens > 0:
            if (
                state_indices_tensor_p is None
                or query_start_loc_p is None
                or first_scheduled_p is None
                or computed_tokens_p is None
            ):
                raise RuntimeError("R002aa checkpoint prefill metadata is incomplete")
            bx_prefill = (b_prefill * x_prefill).transpose(0, 1)
            convolved = causal_conv1d_fn(
                bx_prefill,
                conv_weights,
                self.conv.bias,
                activation=None,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                query_start_loc=query_start_loc_p,
                block_idx_first_scheduled_token=first_scheduled_p,
                block_idx_last_scheduled_token=last_scheduled_p,
                initial_state_idx=last_computed_p,
                num_computed_tokens=computed_tokens_p,
                block_size_to_align=cache_config.block_size,
                metadata=attn_metadata,
            ).transpose(0, 1)[:num_prefill_tokens]
            conv_outputs.append(c_prefill * convolved)

        if num_decode_tokens > 0:
            if state_indices_tensor_d is None:
                raise RuntimeError("R002aa all-mode decode metadata is incomplete")
            bx_decode = (b_decode * x_decode).contiguous()
            if is_all_mode:
                convolved = causal_conv1d_update(
                    bx_decode,
                    conv_state,
                    conv_weights,
                    self.conv.bias,
                    activation=None,
                    conv_state_indices=state_indices_tensor_d,
                    block_idx_last_scheduled_token=last_scheduled_d,
                    initial_state_idx=last_computed_d,
                )
            else:
                convolved = causal_conv1d_update(
                    bx_decode,
                    conv_state,
                    conv_weights,
                    self.conv.bias,
                    activation=None,
                    conv_state_indices=state_indices_tensor_d,
                )
            conv_outputs.insert(0, c_decode * convolved)

        merged = torch.vstack(conv_outputs)
        output[:num_actual_tokens], _ = self.out_proj(merged)
        return None

    base.forward_cuda = forward_cuda_all_mode
    base._r002aa_all_mode_patched = True
    print("R002AA_ALL_SHORTCONV shortconv_patched=1", file=sys.stderr, flush=True)


def _patch_sampler(module: ModuleType) -> None:
    from fused_greedy_head import _patch_sampler as patch_r002k_sampler

    patch_r002k_sampler(module)


def _patch_runner(module: ModuleType) -> None:
    from fused_greedy_head import _patch_runner as patch_r002k_runner

    patch_r002k_runner(module)
    profile_env = (
        "R002Y_PROFILE_ARM",
        "R002Y_PROFILE_FINALIZE",
        "R002Y_PROFILE_OUTPUT_ROOT",
    )
    if all(os.environ.get(name) for name in profile_env):
        from prefill_profiler import _patch_runner as patch_r002y_runner

        patch_r002y_runner(module)
    base = getattr(module, "GPUModelRunner", None)
    if base is None or getattr(base, "_r002aa_prefill_padding_patched", False):
        return

    native_pad = base._pad_for_sequence_parallelism
    multiple = int(os.environ.get("R002AA_PREFILL_PAD_MULTIPLE", "0"))
    minimum_tokens = int(
        os.environ.get("R002AA_PREFILL_PAD_MIN_TOKENS", "4097")
    )
    if multiple < 0 or minimum_tokens < 1:
        raise ValueError("R002aa prefill padding settings must be non-negative")

    @functools.wraps(native_pad)
    def pad_sparse_prefill(self: Any, num_scheduled_tokens: int) -> int:
        native_tokens = native_pad(self, num_scheduled_tokens)
        if (
            not _sparse_enabled()
            or multiple <= 1
            or native_tokens < minimum_tokens
        ):
            return native_tokens
        return ((native_tokens + multiple - 1) // multiple) * multiple

    base._pad_for_sequence_parallelism = pad_sparse_prefill
    base._r002aa_prefill_padding_patched = True
    print(
        "R002AA_PREFILL_PADDING runner_patched=1 "
        f"multiple={multiple} minimum_tokens={minimum_tokens}",
        file=sys.stderr,
        flush=True,
    )


PATCHERS: dict[str, Callable[[ModuleType], None]] = {
    MODEL_CONFIG_MODULE: _patch_model_config,
    MAMBA_METADATA_MODULE: _patch_mamba_metadata,
    MAMBA_MANAGER_MODULE: _patch_mamba_manager,
    SCHEDULER_MODULE: _patch_scheduler,
    LFM2_MODULE: _patch_lfm2,
    SHORT_CONV_MODULE: _patch_short_conv,
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
        module = sys.modules.get(fullname)
        if module is not None:
            patcher(module)
    print(
        "R002AA_RUNTIME "
        f"sparse_enabled={int(_sparse_enabled())} "
        f"mode_file={int(bool(_SPARSE_MODE_FILE))}",
        file=sys.stderr,
        flush=True,
    )
    print("R002AA_ALL_SHORTCONV hook_installed=1", file=sys.stderr, flush=True)
