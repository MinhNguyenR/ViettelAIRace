"""Install the composite R003 research hook."""

from __future__ import annotations

import sys


try:
    from windowed_attention import install_import_hook

    install_import_hook()
except Exception as exc:  # fail at startup; do not silently benchmark native
    print(f"R003_WINDOW hook_install_failed error={exc!r}", file=sys.stderr, flush=True)
    raise
