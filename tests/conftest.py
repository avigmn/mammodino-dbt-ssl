"""Ensure `dbt_ssl` (sibling) and `mammodino_ssl` are importable without pip install."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DBT_SRC = _ROOT.parent / "dbt_simclr_project" / "src"
for p in (_ROOT / "src", _DBT_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
