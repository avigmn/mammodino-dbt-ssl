"""Manifest slice-order lookup (pyarrow-only — avoids pandas env issues)."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

_SLICE_ORDER_COLUMN_PRIORITY: tuple[str, ...] = (
    "slice_index",
    "instance_number",
    "InstanceNumber",
    "z_index",
    "z_rank",
)


def fallback_sort_key_from_abspath(path: str) -> int | None:
    """When manifest has no numeric order column: parse basename stem."""
    import re

    stem = Path(path).stem
    tail = re.search(r"(?:^|[^0-9A-Za-z])(\d+)\s*$", stem)
    if tail:
        return int(tail.group(1))
    found = re.findall(r"\d+", stem)
    if found:
        return int(found[-1])
    return None


def detect_slice_order_column(columns: list[str]) -> tuple[str | None, list[str]]:
    cols_lower = {c.lower(): c for c in columns}
    hits = []
    for want in _SLICE_ORDER_COLUMN_PRIORITY:
        key = want.lower()
        if key in cols_lower:
            hits.append(cols_lower[key])
    return (hits[0] if hits else None), hits


def build_slice_sort_lookup(
    manifest_path: Path,
    *,
    order_column_override: str | None = None,
) -> tuple[dict[str, int], dict[str, object]]:
    """Map ``slice_abspath`` -> int sort key for within-patient ordering."""
    table = pq.read_table(manifest_path)
    cols = table.column_names
    if "slice_abspath" not in cols:
        raise ValueError(f"manifest missing slice_abspath column; got {cols}")

    chosen: str | None = None
    candidates: list[str] = []
    if order_column_override:
        if order_column_override not in cols:
            raise ValueError(f"--slice-order-column {order_column_override!r} not in manifest columns: {cols}")
        chosen = order_column_override
        candidates = [order_column_override]
    else:
        chosen, candidates = detect_slice_order_column(cols)

    provenance: dict[str, object] = {
        "order_column_candidates_found": candidates,
        "order_column_used": chosen,
        "fallback_rule": None,
        "manifest_path": str(manifest_path),
    }

    paths_py = [str(x) for x in table.column("slice_abspath").to_pylist()]
    mapping: dict[str, int] = {}

    if chosen is not None:
        vals_py = table.column(chosen).to_pylist()
        for ap, val in zip(paths_py, vals_py):
            try:
                mapping[ap] = int(val)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Non-integer slice order in column {chosen!r} for {ap}: {val!r}") from e
        return mapping, provenance

    provenance["fallback_rule"] = "basename_digits_else_lex_rank_offset"
    ap_unique = sorted(set(paths_py))
    lex_need = [ap for ap in ap_unique if fallback_sort_key_from_abspath(ap) is None]
    lex_rank = {ap: i for i, ap in enumerate(sorted(set(lex_need)))}
    for ap in ap_unique:
        k = fallback_sort_key_from_abspath(ap)
        mapping[ap] = int(k) if k is not None else int(1_000_000_000 + lex_rank[ap])

    return mapping, provenance
