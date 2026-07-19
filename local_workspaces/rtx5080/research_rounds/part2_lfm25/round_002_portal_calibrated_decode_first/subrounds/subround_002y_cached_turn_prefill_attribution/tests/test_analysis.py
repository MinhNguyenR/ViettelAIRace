#!/usr/bin/env python3
"""Deterministic helper tests for R002y attribution."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_prefill_panel.py"
)
SPEC = importlib.util.spec_from_file_location("r002y_analyze", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    assert MODULE.kernel_family("flash_attn_fwd_kernel") == "attention"
    assert MODULE.kernel_family("f8f8bf16_rowwise") == "fp8_gemm"
    assert MODULE.kernel_family("causal_conv1d_update_kernel") == "short_conv"
    assert MODULE.kernel_family("partial_greedy_kernel") == "r002k_output_head"
    assert MODULE.kernel_family("memcpy32_post") == "copy"
    print("r002y_analysis_tests=passed")


if __name__ == "__main__":
    main()
