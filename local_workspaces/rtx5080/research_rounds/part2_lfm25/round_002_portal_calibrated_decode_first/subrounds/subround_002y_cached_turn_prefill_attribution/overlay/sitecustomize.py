"""Install the frozen R002k serving path and R002y profiler."""

from __future__ import annotations

import sys


try:
    from fused_greedy_head import install_import_hook as install_r002k
    from prefill_profiler import install_import_hook as install_r002y

    install_r002k()
    install_r002y()
except Exception as exc:
    print(
        f"R002Y_PROFILE hook_install_failed error={exc!r}",
        file=sys.stderr,
        flush=True,
    )
