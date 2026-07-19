"""Install the frozen serving path and R002x profiler."""

from __future__ import annotations

import os
import sys


try:
    from fused_greedy_head import install_import_hook as install_r002k

    install_r002k()
    if os.environ.get("R002X_ENABLE_R002W") == "1":
        from short_conv_decode_fusion import install_import_hook as install_r002w

        install_r002w()
    from graph_step_profiler import install_import_hook as install_r002x

    install_r002x()
except Exception as exc:
    print(
        f"R002X_PROFILE hook_install_failed error={exc!r}",
        file=sys.stderr,
        flush=True,
    )
