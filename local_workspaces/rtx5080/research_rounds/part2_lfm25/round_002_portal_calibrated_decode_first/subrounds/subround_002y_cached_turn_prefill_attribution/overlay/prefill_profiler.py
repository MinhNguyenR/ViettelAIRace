"""Profile one isolated prefill-to-first-token boundary in vLLM."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor-r002y")

import torch


RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
MAX_PROFILE_STEPS = 32
_HOOK_INSTALLED = False


@dataclass(frozen=True)
class Command:
    panel_id: str
    expected_prompt_tokens: int
    kineto: bool


@dataclass(frozen=True)
class RequestStep:
    req_id: str
    is_new: bool
    prompt_tokens: int | None
    computed_before: int
    scheduled_tokens: int


class _State:
    active = False
    pending_finish = False
    last_panel_id: str | None = None
    command: Command | None = None
    req_id: str | None = None
    prompt_tokens = 0
    initial_computed_tokens = 0
    profiler: Any = None
    boundary_start: Any = None
    boundary_end: Any = None
    step_starts: list[Any] = []
    step_ends: list[Any] = []
    step_rows: list[dict[str, Any]] = []
    cpu_step_ms: list[float] = []
    step_cpu_started_ns = 0
    step_inflight = False
    terminal_step = False
    started_ns = 0


STATE = _State()


def _output_root() -> Path:
    return Path(os.environ["R002Y_PROFILE_OUTPUT_ROOT"])


def _arm_path() -> Path:
    return Path(os.environ["R002Y_PROFILE_ARM"])


def _finalize_path() -> Path:
    return Path(os.environ["R002Y_PROFILE_FINALIZE"])


def _read_command() -> Command | None:
    path = _arm_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        command = Command(
            panel_id=str(payload["panel_id"]),
            expected_prompt_tokens=int(payload["expected_prompt_tokens"]),
            kineto=bool(payload.get("kineto", False)),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    normalized_panel_id = command.panel_id.replace("-", "").replace("_", "")
    if not normalized_panel_id.isalnum():
        raise RuntimeError(f"invalid R002y panel id: {command.panel_id!r}")
    if not 2 <= command.expected_prompt_tokens <= 32768:
        raise RuntimeError(
            f"invalid R002y prompt length: {command.expected_prompt_tokens}"
        )
    if command.panel_id == STATE.last_panel_id:
        return None
    return command


def _request_steps(scheduler_output: Any) -> list[RequestStep]:
    scheduled = {
        str(req_id): int(count)
        for req_id, count in scheduler_output.num_scheduled_tokens.items()
    }
    rows: list[RequestStep] = []
    for request in scheduler_output.scheduled_new_reqs:
        req_id = str(request.req_id)
        prompt_ids = request.prompt_token_ids
        if prompt_ids is None:
            prompt_ids = request.prefill_token_ids
        rows.append(
            RequestStep(
                req_id=req_id,
                is_new=True,
                prompt_tokens=len(prompt_ids) if prompt_ids is not None else None,
                computed_before=int(request.num_computed_tokens),
                scheduled_tokens=scheduled[req_id],
            )
        )

    cached = scheduler_output.scheduled_cached_reqs
    for index, req_id_raw in enumerate(cached.req_ids):
        req_id = str(req_id_raw)
        rows.append(
            RequestStep(
                req_id=req_id,
                is_new=False,
                prompt_tokens=None,
                computed_before=int(cached.num_computed_tokens[index]),
                scheduled_tokens=scheduled[req_id],
            )
        )
    return rows


def _start(command: Command, row: RequestStep) -> None:
    if row.prompt_tokens != command.expected_prompt_tokens:
        raise RuntimeError(
            "R002y prompt mismatch: "
            f"expected={command.expected_prompt_tokens} observed={row.prompt_tokens}"
        )
    STATE.command = command
    STATE.req_id = row.req_id
    STATE.prompt_tokens = int(row.prompt_tokens)
    STATE.initial_computed_tokens = row.computed_before
    STATE.boundary_start = torch.cuda.Event(enable_timing=True)
    STATE.boundary_end = torch.cuda.Event(enable_timing=True)
    STATE.step_starts = [
        torch.cuda.Event(enable_timing=True) for _ in range(MAX_PROFILE_STEPS)
    ]
    STATE.step_ends = [
        torch.cuda.Event(enable_timing=True) for _ in range(MAX_PROFILE_STEPS)
    ]
    STATE.step_rows = []
    STATE.cpu_step_ms = []
    STATE.step_inflight = False
    STATE.terminal_step = False
    STATE.started_ns = time.perf_counter_ns()
    STATE.profiler = None
    if command.kineto:
        activities = torch.profiler.supported_activities()
        if torch.profiler.ProfilerActivity.CUDA not in activities:
            raise RuntimeError(f"CUDA profiler activity unavailable: {activities}")
        STATE.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            acc_events=True,
        )
        STATE.profiler.__enter__()
    STATE.boundary_start.record()
    STATE.active = True
    STATE.pending_finish = False
    print(
        "R002Y_PROFILE active=1 "
        f"panel={command.panel_id} req={row.req_id} "
        f"prompt={row.prompt_tokens} computed={row.computed_before} "
        f"kineto={int(command.kineto)}",
        file=sys.stderr,
        flush=True,
    )


def _finish() -> None:
    command = STATE.command
    if command is None or STATE.boundary_start is None or STATE.boundary_end is None:
        raise RuntimeError("R002y finalize without a completed panel")
    torch.cuda.synchronize()
    boundary_gpu_ms = float(STATE.boundary_start.elapsed_time(STATE.boundary_end))
    step_gpu_ms = [
        float(STATE.step_starts[index].elapsed_time(STATE.step_ends[index]))
        for index in range(len(STATE.step_rows))
    ]

    panel_dir = _output_root() / "panels" / command.panel_id
    panel_dir.mkdir(parents=True, exist_ok=True)
    trace_path: str | None = None
    if STATE.profiler is not None:
        STATE.profiler.__exit__(None, None, None)
        output = panel_dir / "trace.json"
        STATE.profiler.export_chrome_trace(str(output))
        trace_path = str(output)

    scheduled_total = sum(int(row["scheduled_tokens"]) for row in STATE.step_rows)
    metadata = {
        "schema": "r002y.prefill_profile.v1",
        "panel_id": command.panel_id,
        "valid": (
            bool(STATE.step_rows)
            and STATE.initial_computed_tokens + scheduled_total
            >= STATE.prompt_tokens
        ),
        "kineto": command.kineto,
        "req_id": STATE.req_id,
        "prompt_tokens": STATE.prompt_tokens,
        "initial_computed_tokens": STATE.initial_computed_tokens,
        "scheduled_prefill_tokens": scheduled_total,
        "cache_fraction": STATE.initial_computed_tokens / STATE.prompt_tokens,
        "step_count": len(STATE.step_rows),
        "step_rows": STATE.step_rows,
        "boundary_gpu_ms": boundary_gpu_ms,
        "step_gpu_ms": step_gpu_ms,
        "cpu_step_ms": STATE.cpu_step_ms,
        "profile_wall_ms": (time.perf_counter_ns() - STATE.started_ns) / 1_000_000,
        "per_step_synchronize": False,
        "finalized_after_target_response": True,
        "trace_path": trace_path,
    }
    (panel_dir / "profile_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (panel_dir / "done.json").write_text(
        json.dumps(
            {"panel_id": command.panel_id, "valid": metadata["valid"]}, indent=2
        )
        + "\n"
    )
    try:
        current = json.loads(_arm_path().read_text())
        if str(current.get("panel_id")) == command.panel_id:
            _arm_path().unlink(missing_ok=True)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    _finalize_path().unlink(missing_ok=True)

    STATE.active = False
    STATE.pending_finish = False
    STATE.last_panel_id = command.panel_id
    STATE.command = None
    STATE.req_id = None
    STATE.profiler = None
    STATE.step_inflight = False
    print(
        "R002Y_PROFILE active=0 done=1 "
        f"panel={command.panel_id} steps={metadata['step_count']} "
        f"gpu_ms={boundary_gpu_ms:.6f} valid={int(metadata['valid'])}",
        file=sys.stderr,
        flush=True,
    )


def _patch_runner(module: ModuleType) -> None:
    from fused_greedy_head import PATCHERS

    PATCHERS[RUNNER_MODULE](module)
    base = getattr(module, "GPUModelRunner", None)
    if base is None or getattr(base, "_r002y_profile_patched", False):
        return
    native_execute_model = base.execute_model
    native_sample_tokens = base.sample_tokens

    @functools.wraps(native_execute_model)
    def execute_model_profiled(
        self: Any,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
    ) -> Any:
        if STATE.pending_finish and _finalize_path().exists():
            _finish()

        rows = _request_steps(scheduler_output)
        if not STATE.active and not STATE.pending_finish:
            command = _read_command()
            if command is not None:
                candidates = [
                    row
                    for row in rows
                    if row.is_new
                    and row.prompt_tokens == command.expected_prompt_tokens
                ]
                if len(rows) != 1 or len(candidates) != 1:
                    raise RuntimeError(
                        "R002y target request was not isolated: "
                        f"rows={len(rows)} candidates={len(candidates)}"
                    )
                _start(command, candidates[0])

        if STATE.active:
            target = [row for row in rows if row.req_id == STATE.req_id]
            if len(target) != 1:
                raise RuntimeError(
                    f"R002y target missing or duplicated: req={STATE.req_id} rows={rows}"
                )
            if len(STATE.step_rows) >= MAX_PROFILE_STEPS:
                raise RuntimeError("R002y exceeded maximum prefill chunks")
            row = target[0]
            step = len(STATE.step_rows)
            STATE.step_rows.append(
                {
                    "step": step,
                    "is_new": row.is_new,
                    "computed_before": row.computed_before,
                    "scheduled_tokens": row.scheduled_tokens,
                }
            )
            STATE.step_starts[step].record()
            STATE.step_cpu_started_ns = time.perf_counter_ns()
            STATE.step_inflight = True
            STATE.terminal_step = (
                row.computed_before + row.scheduled_tokens >= STATE.prompt_tokens
            )

        return native_execute_model(self, scheduler_output, intermediate_tensors)

    @functools.wraps(native_sample_tokens)
    def sample_tokens_profiled(self: Any, grammar_output: Any) -> Any:
        result = native_sample_tokens(self, grammar_output)
        if STATE.active and STATE.step_inflight:
            step = len(STATE.step_rows) - 1
            STATE.step_ends[step].record()
            STATE.cpu_step_ms.append(
                (time.perf_counter_ns() - STATE.step_cpu_started_ns) / 1_000_000
            )
            if STATE.profiler is not None:
                STATE.profiler.step()
            STATE.step_inflight = False
            if STATE.terminal_step:
                STATE.boundary_end.record()
                STATE.active = False
                STATE.pending_finish = True
        return result

    base.execute_model = execute_model_profiled
    base.sample_tokens = sample_tokens_profiled
    base._r002y_profile_patched = True
    print("R002Y_PROFILE runner_patched=1", file=sys.stderr, flush=True)


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
    print("R002Y_PROFILE hook_installed=1", file=sys.stderr, flush=True)
