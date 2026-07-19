"""Install the composite R003b selective-window research hook."""

from __future__ import annotations

import sys


try:
    from selective_global_attention import install_import_hook

    install_import_hook()
except Exception as exc:  # fail at startup; never benchmark a silent fallback
    print(f"R003B_GLOBAL hook_install_failed error={exc!r}", file=sys.stderr, flush=True)
    raise
