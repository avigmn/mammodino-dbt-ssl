#!/usr/bin/env python3
"""
Toy DINO (phase 1: image-level loss only) on CIFAR-10 + linear probe.

Expects sibling repo `dbt_simclr_project` for RobustCIFAR10, or install `dbt-ssl` editable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_DBT_SRC = _REPO.parent / "dbt_simclr_project" / "src"
for p in (_REPO / "src", _DBT_SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mammodino_ssl.models.dino_ssl import create_dino_ssl
from mammodino_ssl.models.linear_probe import FrozenTinyViTLinearProbe
from mammodino_ssl.toy.cifar_dino_dataset import CIFAR10DINOPairDataset, CIFAR10SupervisedEvalDataset
from mammodino_ssl.train.dino_loss import DINOLogitCenter
from mammodino_ssl.train.dino_trainer import DINOTrainer, collate_dino_views


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_supervised(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    x = torch.stack([b["image"] for b in batch], dim=0)
    y = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"image": x, "label": y}


def _plot_dino_losses(history: list[dict[str, Any]], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train_dino")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val_dino")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DINO loss")
    ax.legend()
    ax.set_title("Toy DINO (CIFAR-10)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_probe_metrics(history: list[dict[str, Any]], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].set_title("Linear probe loss")

    axes[1].plot(epochs, [h["train_acc"] for h in history], marker="o", label="train_acc")
    axes[1].plot(epochs, [h["val_acc"] for h in history], marker="o", label="val_acc")
    axes[1].axhline(0.1, color="gray", linestyle="--", label="random (10%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].set_title("Linear probe (10-class)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def _eval_probe_accuracy_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
) -> tuple[float, float]:
    model.eval()
    amp_on = amp and device.type == "cuda"
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model(x)
            loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return total_loss / max(1, total), correct / max(1, total)


def _run_linear_probe(
    *,
    ckpt_path: Path,
    data_root: Path,
    out_dir: Path,
    device: torch.device,
    probe_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    pin_memory: bool,
    amp: bool,
    image_size: int,
    num_prototypes: int,
) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = create_dino_ssl(image_size=image_size, num_prototypes=num_prototypes)
    model.load_state_dict(ckpt["model_state"], strict=True)
    probe = FrozenTinyViTLinearProbe(model.student_backbone, num_classes=10).to(device)

    train_ds = CIFAR10SupervisedEvalDataset(str(data_root), train=True, image_size=image_size)
    val_ds = CIFAR10SupervisedEvalDataset(str(data_root), train=False, image_size=image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_supervised,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_supervised,
    )

    opt = AdamW(probe.head.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(probe_epochs):
        probe.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_n = 0
        pbar = tqdm(
            train_loader,
            desc=f"probe {epoch+1}/{probe_epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = probe(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss_sum += float(loss.item()) * y.size(0)
            train_correct += int((logits.argmax(1) == y).sum().item())
            train_n += int(y.numel())

        train_loss = train_loss_sum / max(1, train_n)
        train_acc = train_correct / max(1, train_n)
        val_loss, val_acc = _eval_probe_accuracy_loss(probe, val_loader, device, amp=amp)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"probe epoch={epoch+1} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({"head_state": probe.head.state_dict(), "val_acc": val_acc}, out_dir / "probe_best.pt")

    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "random_baseline_acc": 0.1,
        "beats_random": best_val_acc > 0.15,
    }
    (out_dir / "probe_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_probe_metrics(history, out_dir / "plots" / "probe_curves.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toy DINO (phase 1) + linear probe on CIFAR-10.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples (run the script directly — do not prefix with pytest):\n"
            "  python scripts/run_toy_dino_cifar10.py --epochs 3 --probe-epochs 2 --batch-size 32\n"
            "  PYTHONUNBUFFERED=1 python scripts/run_toy_dino_cifar10.py --num-workers 0\n"
            "If CUDA driver errors appear, use:  --device cpu --no-amp\n"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30, help="DINO SSL epochs")
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-prototypes", type=int, default=512, help="DINO prototype count K")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-probe", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _set_seed(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir if args.out_dir is not None else _REPO / "experiments" / "toy_dino_cifar10" / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = out_dir / "cifar10_data"
    data_root.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    amp = not args.no_amp

    train_ds = CIFAR10DINOPairDataset(str(data_root), train=True, image_size=args.image_size)
    val_ds = CIFAR10DINOPairDataset(str(data_root), train=False, image_size=args.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_dino_views,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_dino_views,
        drop_last=False,
    )

    model = create_dino_ssl(image_size=args.image_size, num_prototypes=args.num_prototypes)
    center = DINOLogitCenter(args.num_prototypes, center_momentum=args.center_momentum)
    ckpt_dir = out_dir / "dino_checkpoints"
    log_dir = out_dir / "dino_logs"
    trainer = DINOTrainer(
        model=model,
        center=center,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        teacher_momentum=args.teacher_momentum,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        grad_accum_steps=args.grad_accum,
        amp=amp,
        checkpoints_dir=ckpt_dir,
        logs_dir=log_dir,
    )
    print(f"Output directory: {out_dir}")
    dino_out = trainer.fit(epochs=args.epochs, max_train_steps=None, max_val_steps=None)
    print(f"DINO best_val_loss={dino_out.best_val_loss:.4f} best_epoch={dino_out.best_epoch}")
    _plot_dino_losses(
        json.loads(dino_out.history_path.read_text(encoding="utf-8")),
        out_dir / "plots" / "dino_loss.png",
    )

    summary = _run_linear_probe(
        ckpt_path=dino_out.best_checkpoint,
        data_root=data_root,
        out_dir=out_dir,
        device=device,
        probe_epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        lr=args.lr_probe,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        amp=amp,
        image_size=args.image_size,
        num_prototypes=args.num_prototypes,
    )
    print(json.dumps(summary, indent=2))
    (out_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir.resolve()),
                "dino_best_val_loss": dino_out.best_val_loss,
                "dino_best_epoch": dino_out.best_epoch,
                "linear_probe": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
