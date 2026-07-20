"""Install the composed R002k, R002y and R002aa import overlay."""

from __future__ import annotations

import sys


try:
    from lfm2_all_shortconv import install_import_hook

    install_import_hook()
except Exception as exc:
    print(
        f"R002AA_ALL_SHORTCONV hook_install_failed error={exc!r}",
        file=sys.stderr,
        flush=True,
    )
