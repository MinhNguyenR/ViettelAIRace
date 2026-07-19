"""Install the composite R003d tiered-window research hook."""

from __future__ import annotations

import sys


try:
    from tiered_attention import install_import_hook

    install_import_hook()
except Exception as exc:
    print(f"R003D_TIERED hook_install_failed error={exc!r}", file=sys.stderr, flush=True)
    raise
