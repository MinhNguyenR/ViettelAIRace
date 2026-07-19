"""Profile complete vLLM decode graph steps without per-step synchronization."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor-r002x")

import torch


RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
_HOOK_INSTALLED = False


@dataclass(frozen=True)
class Command:
    panel_id: str
    target_batch: int
    target_steps: int


class _State:
    active = False
    last_panel_id: str | None = None
    command: Command | None = None
    profiler: Any = None
    completed_steps = 0
    started_ns = 0
    step_cpu_started_ns = 0
    step_inflight = False
    start_events: list[Any] = []
    end_events: list[Any] = []
    cpu_step_ms: list[float] = []
    observed_batches: list[int] = []
    observed_token_counts: list[list[int]] = []


STATE = _State()


class _Telemetry:
    total_steps = 0
    pure_decode_steps = 0
    mixed_steps = 0
    prefill_only_steps = 0
    empty_steps = 0
    pure_decode_rows = 0
    mixed_decode_rows = 0
    prefill_requests = 0
    prefill_tokens = 0
    scheduled_requests = 0
    scheduled_tokens = 0
    all_batch_histogram: Counter[int] = Counter()
    pure_decode_batch_histogram: Counter[int] = Counter()
    mixed_decode_rows_histogram: Counter[int] = Counter()


TELEMETRY = _Telemetry()


def _output_root() -> Path:
    return Path(os.environ["R002X_PROFILE_OUTPUT_ROOT"])


def _arm_path() -> Path:
    return Path(os.environ["R002X_PROFILE_ARM"])


def _telemetry_enabled() -> bool:
    return os.environ.get("R002X_TELEMETRY") == "1"


def _telemetry_output_path() -> Path:
    return Path(os.environ["R002X_TELEMETRY_OUTPUT"])


def _telemetry_dump_path() -> Path:
    return Path(os.environ["R002X_TELEMETRY_DUMP"])


def _decode_counts(scheduler_output: Any) -> list[int]:
    values = getattr(scheduler_output, "num_scheduled_tokens", {}).values()
    return sorted(int(value) for value in values)


def _write_telemetry(*, final: bool) -> None:
    path = _telemetry_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "r002x.scheduler_telemetry.v1",
        "final": final,
        "classification": (
            "count==1 is treated as decode; count>1 as prefill/chunk"
        ),
        "total_steps": TELEMETRY.total_steps,
        "pure_decode_steps": TELEMETRY.pure_decode_steps,
        "mixed_steps": TELEMETRY.mixed_steps,
        "prefill_only_steps": TELEMETRY.prefill_only_steps,
        "empty_steps": TELEMETRY.empty_steps,
        "pure_decode_rows": TELEMETRY.pure_decode_rows,
        "mixed_decode_rows": TELEMETRY.mixed_decode_rows,
        "prefill_requests": TELEMETRY.prefill_requests,
        "prefill_tokens": TELEMETRY.prefill_tokens,
        "scheduled_requests": TELEMETRY.scheduled_requests,
        "scheduled_tokens": TELEMETRY.scheduled_tokens,
        "all_batch_histogram": {
            str(key): value for key, value in sorted(TELEMETRY.all_batch_histogram.items())
        },
        "pure_decode_batch_histogram": {
            str(key): value
            for key, value in sorted(TELEMETRY.pure_decode_batch_histogram.items())
        },
        "mixed_decode_rows_histogram": {
            str(key): value
            for key, value in sorted(TELEMETRY.mixed_decode_rows_histogram.items())
        },
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _maybe_dump_telemetry() -> None:
    path = _telemetry_dump_path()
    if not path.exists():
        return
    _write_telemetry(final=True)
    path.unlink(missing_ok=True)
    print(
        "R002X_TELEMETRY final=1 "
        f"steps={TELEMETRY.total_steps} pure_decode={TELEMETRY.pure_decode_steps} "
        f"mixed={TELEMETRY.mixed_steps} prefill={TELEMETRY.prefill_only_steps}",
        file=sys.stderr,
        flush=True,
    )


def _record_telemetry(scheduler_output: Any) -> None:
    counts = _decode_counts(scheduler_output)
    TELEMETRY.total_steps += 1
    TELEMETRY.scheduled_requests += len(counts)
    TELEMETRY.scheduled_tokens += sum(counts)
    TELEMETRY.all_batch_histogram[len(counts)] += 1
    if not counts:
        TELEMETRY.empty_steps += 1
    elif all(count == 1 for count in counts):
        TELEMETRY.pure_decode_steps += 1
        TELEMETRY.pure_decode_rows += len(counts)
        TELEMETRY.pure_decode_batch_histogram[len(counts)] += 1
    elif all(count > 1 for count in counts):
        TELEMETRY.prefill_only_steps += 1
        TELEMETRY.prefill_requests += len(counts)
        TELEMETRY.prefill_tokens += sum(counts)
    else:
        decode_rows = sum(count == 1 for count in counts)
        prefill_counts = [count for count in counts if count > 1]
        TELEMETRY.mixed_steps += 1
        TELEMETRY.mixed_decode_rows += decode_rows
        TELEMETRY.prefill_requests += len(prefill_counts)
        TELEMETRY.prefill_tokens += sum(prefill_counts)
        TELEMETRY.mixed_decode_rows_histogram[decode_rows] += 1
    if TELEMETRY.total_steps % 512 == 0:
        _write_telemetry(final=False)


def _read_command() -> Command | None:
    path = _arm_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        command = Command(
            panel_id=str(payload["panel_id"]),
            target_batch=int(payload["target_batch"]),
            target_steps=int(payload["target_steps"]),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if not command.panel_id.replace("-", "_").isalnum():
        raise RuntimeError(f"invalid R002x panel id: {command.panel_id!r}")
    if command.target_batch not in {1, 2, 4, 8}:
        raise RuntimeError(f"invalid R002x target batch: {command.target_batch}")
    if not 8 <= command.target_steps <= 512:
        raise RuntimeError(f"invalid R002x target steps: {command.target_steps}")
    if command.panel_id == STATE.last_panel_id:
        return None
    return command


def _is_target_decode(command: Command, scheduler_output: Any) -> bool:
    counts = _decode_counts(scheduler_output)
    return len(counts) == command.target_batch and all(count == 1 for count in counts)


def _start_profiler(command: Command) -> None:
    activities = torch.profiler.supported_activities()
    if torch.profiler.ProfilerActivity.CUDA not in activities:
        raise RuntimeError(f"CUDA profiler activity unavailable: {activities}")

    STATE.start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(command.target_steps)
    ]
    STATE.end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(command.target_steps)
    ]
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        acc_events=True,
    )
    profiler.__enter__()
    STATE.active = True
    STATE.command = command
    STATE.profiler = profiler
    STATE.completed_steps = 0
    STATE.started_ns = time.perf_counter_ns()
    STATE.step_inflight = False
    STATE.cpu_step_ms = []
    STATE.observed_batches = []
    STATE.observed_token_counts = []
    print(
        "R002X_PROFILE active=1 "
        f"panel={command.panel_id} batch={command.target_batch} "
        f"steps={command.target_steps}",
        file=sys.stderr,
        flush=True,
    )


def _finish_profiler() -> None:
    command = STATE.command
    profiler = STATE.profiler
    if command is None or profiler is None:
        raise RuntimeError("R002x profiler finished without active command")

    torch.cuda.synchronize()
    gpu_step_ms = [
        float(start.elapsed_time(end))
        for start, end in zip(STATE.start_events, STATE.end_events, strict=True)
    ]
    profiler.__exit__(None, None, None)

    panel_dir = _output_root() / "panels" / command.panel_id
    panel_dir.mkdir(parents=True, exist_ok=True)
    trace_path = panel_dir / "trace.json"
    metadata_path = panel_dir / "profile_metadata.json"
    done_path = panel_dir / "done.json"
    profiler.export_chrome_trace(str(trace_path))

    all_decode_only = all(
        counts and all(count == 1 for count in counts)
        for counts in STATE.observed_token_counts
    )
    all_target_batch = all(
        batch == command.target_batch for batch in STATE.observed_batches
    )
    metadata = {
        "panel_id": command.panel_id,
        "target_batch": command.target_batch,
        "target_steps": command.target_steps,
        "completed_steps": STATE.completed_steps,
        "observed_batches": STATE.observed_batches,
        "observed_token_counts": STATE.observed_token_counts,
        "all_decode_only": all_decode_only,
        "all_target_batch": all_target_batch,
        "gpu_step_ms": gpu_step_ms,
        "cpu_step_ms": STATE.cpu_step_ms,
        "panel_wall_ms": (time.perf_counter_ns() - STATE.started_ns) / 1_000_000,
        "trace_path": str(trace_path),
        "per_step_synchronize": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    done_path.write_text(
        json.dumps(
            {
                "panel_id": command.panel_id,
                "valid": (
                    STATE.completed_steps == command.target_steps
                    and all_decode_only
                    and all_target_batch
                ),
            },
            indent=2,
        )
        + "\n"
    )
    try:
        current = json.loads(_arm_path().read_text())
        if str(current.get("panel_id")) == command.panel_id:
            _arm_path().unlink(missing_ok=True)
    except (OSError, ValueError, json.JSONDecodeError):
        pass

    STATE.active = False
    STATE.last_panel_id = command.panel_id
    STATE.command = None
    STATE.profiler = None
    STATE.step_inflight = False
    print(
        "R002X_PROFILE active=0 done=1 "
        f"panel={command.panel_id} steps={STATE.completed_steps} "
        f"valid={int(all_decode_only and all_target_batch)}",
        file=sys.stderr,
        flush=True,
    )


def _patch_runner(module: ModuleType) -> None:
    from fused_greedy_head import PATCHERS

    PATCHERS[RUNNER_MODULE](module)
    base = getattr(module, "GPUModelRunner", None)
    if base is None or getattr(base, "_r002x_profile_patched", False):
        return
    native_execute_model = base.execute_model
    native_sample_tokens = base.sample_tokens

    @functools.wraps(native_execute_model)
    def execute_model_profiled(
        self: Any,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
    ) -> Any:
        if _telemetry_enabled():
            _maybe_dump_telemetry()
            _record_telemetry(scheduler_output)
        if not STATE.active:
            command = _read_command()
            if command is not None and _is_target_decode(command, scheduler_output):
                _start_profiler(command)

        if STATE.active:
            command = STATE.command
            assert command is not None
            counts = _decode_counts(scheduler_output)
            STATE.observed_batches.append(len(counts))
            STATE.observed_token_counts.append(counts)
            step = STATE.completed_steps
            if step >= command.target_steps:
                raise RuntimeError("R002x observed an extra active profile step")
            STATE.start_events[step].record()
            STATE.step_cpu_started_ns = time.perf_counter_ns()
            STATE.step_inflight = True

        return native_execute_model(self, scheduler_output, intermediate_tensors)

    @functools.wraps(native_sample_tokens)
    def sample_tokens_profiled(self: Any, grammar_output: Any) -> Any:
        result = native_sample_tokens(self, grammar_output)
        if STATE.active and STATE.step_inflight:
            step = STATE.completed_steps
            STATE.end_events[step].record()
            STATE.cpu_step_ms.append(
                (time.perf_counter_ns() - STATE.step_cpu_started_ns) / 1_000_000
            )
            STATE.profiler.step()
            STATE.completed_steps += 1
            STATE.step_inflight = False
            command = STATE.command
            assert command is not None
            if STATE.completed_steps >= command.target_steps:
                _finish_profiler()
        return result

    base.execute_model = execute_model_profiled
    base.sample_tokens = sample_tokens_profiled
    base._r002x_profile_patched = True
    print("R002X_PROFILE runner_patched=1", file=sys.stderr, flush=True)


class _OverlayLoader(importlib.abc.Loader):
    def __init__(self, original_loader: importlib.abc.Loader) -> None:
        self.original_loader = original_loader

    def create_module(self, spec: Any) -> Any:
        create_module = getattr(self.original_loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self.original_loader.exec_module(module)
        _patch_runner(module)


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != RUNNER_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _OverlayLoader(spec.loader)
        return spec


def install_import_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _OverlayFinder())
    _HOOK_INSTALLED = True
    loaded = sys.modules.get(RUNNER_MODULE)
    if loaded is not None:
        _patch_runner(loaded)
    print("R002X_PROFILE hook_installed=1", file=sys.stderr, flush=True)
