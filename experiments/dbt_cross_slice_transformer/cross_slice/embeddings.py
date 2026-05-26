"""Frozen CLS extraction + per-patient ordering by manifest slice sort key."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _to_tensor3(image_nchw: object) -> torch.Tensor:
    t = torch.from_numpy(image_nchw).float()  # type: ignore[arg-type]
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    return t


def collate_supervised_with_sort(
    batch: list[dict],
    sort_lookup: dict[str, int],
) -> dict[str, object]:
    x = torch.stack([_to_tensor3(item["image"]) for item in batch], dim=0)
    y = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    meta = []
    sort_keys: list[int] = []
    for item in batch:
        ap = str(item["slice_abspath"])
        if ap not in sort_lookup:
            raise KeyError(f"slice_abspath not in manifest lookup: {ap}")
        meta.append({"canonical_patient_id": item.get("canonical_patient_id"), "slice_abspath": ap})
        sort_keys.append(int(sort_lookup[ap]))
    return {
        "image": x,
        "label": y,
        "meta": meta,
        "slice_sort_key": torch.tensor(sort_keys, dtype=torch.long),
    }


@torch.no_grad()
def extract_cls_embeddings(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None,
    desc: str,
    progress: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    backbone.eval()
    amp_on = amp and device.type == "cuda"
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    pids: list[str] = []
    sks: list[np.ndarray] = []

    # tqdm total must reflect early exit (--max-*-batches) or ETA is wildly wrong.
    stream: DataLoader | tqdm
    if progress:
        try:
            n_loader = len(loader)
        except TypeError:
            n_loader = None
        if max_batches is None:
            bar_total: int | None = n_loader
        elif n_loader is None:
            bar_total = int(max_batches)
        else:
            bar_total = min(int(max_batches), int(n_loader))
        stream = tqdm(loader, desc=desc, total=bar_total, leave=True, dynamic_ncols=True)
    else:
        stream = loader

    for step_idx, batch in enumerate(stream):
        if max_batches is not None and step_idx >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        meta = batch["meta"]
        sort_k = batch["slice_sort_key"].detach().cpu().numpy()
        with torch.amp.autocast("cuda", enabled=amp_on):
            cls, _ = backbone(x)
        xs.append(cls.detach().float().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        sks.append(sort_k)
        for m in meta:
            pids.append(str(m["canonical_patient_id"]))
    X = np.concatenate(xs, axis=0) if xs else np.zeros((0, 0), dtype=np.float32)
    Y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64)
    SK = np.concatenate(sks, axis=0) if sks else np.zeros((0,), dtype=np.int64)
    return X, Y, pids, SK


def group_slices_per_patient_ordered(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: list[str],
    sort_keys: np.ndarray,
) -> tuple[list[str], list[np.ndarray], np.ndarray, dict[str, object]]:
    """Stack embeddings per patient sorted by (slice_sort_key, insertion_index)."""
    by_pid: dict[str, dict] = defaultdict(lambda: {"items": [], "label": None})
    for i in range(len(y)):
        pid = patient_ids[i]
        sk = int(sort_keys[i])
        block = by_pid[pid]
        block["items"].append((sk, len(block["items"]), X[i].astype(np.float32)))
        lab = int(y[i])
        if block["label"] is None:
            block["label"] = lab
        elif block["label"] != lab:
            raise ValueError(f"Inconsistent labels for patient {pid}: {block['label']} vs {lab}")

    p_sorted = sorted(by_pid.keys())
    mats: list[np.ndarray] = []
    labels = np.empty(len(p_sorted), dtype=np.int64)
    dup_warn: dict[str, set[int]] = defaultdict(set)
    diag: dict[str, object] = {"duplicate_sort_keys": [], "per_patient_slice_counts": {}}

    for pid in p_sorted:
        items = by_pid[pid]["items"]
        items_sorted = sorted(items, key=lambda t: (t[0], t[1]))
        sk_list = [t[0] for t in items_sorted]
        seen: set[int] = set()
        for sk in sk_list:
            if sk in seen:
                dup_warn[pid].add(sk)
            seen.add(sk)
        mats.append(np.stack([t[2] for t in items_sorted], axis=0))
        labels[len(mats) - 1] = int(by_pid[pid]["label"])
        diag["per_patient_slice_counts"][pid] = len(items_sorted)

    if dup_warn:
        diag["duplicate_sort_keys"] = [{pid: sorted(vs)} for pid, vs in dup_warn.items()]
    return p_sorted, mats, labels, diag
