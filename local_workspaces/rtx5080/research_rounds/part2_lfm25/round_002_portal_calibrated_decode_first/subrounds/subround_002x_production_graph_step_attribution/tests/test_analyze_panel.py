#!/usr/bin/env python3
"""Deterministic kernel-family tests for R002x Kineto attribution."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_panel.py"
SPEC = importlib.util.spec_from_file_location("r002x_analyze", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    assert (
        MODULE.kernel_family("void at::native::binary_internal::mulfunctor")
        == "native_shortconv_mul"
    )
    assert MODULE.kernel_family("memcpy32_post") == "native_shortconv_staging"
    assert (
        MODULE.kernel_family("short_conv_decode_epilogue_kernel")
        == "r002w_fused_epilogue"
    )
    print("r002x_analyze_tests=passed")


if __name__ == "__main__":
    main()
