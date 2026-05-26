"""Minimal DINO trainer (student–teacher, phase 1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mammodino_ssl.models.dino_ssl import DINOSSLModule
from mammodino_ssl.train.dino_loss import DINOLogitCenter, dino_cross_entropy, masked_patch_cross_entropy


@dataclass
class DINOTrainOutput:
    best_val_loss: float
    best_epoch: int
    best_checkpoint: Path
    history_path: Path
    summary_path: Path
    plots_dir: Path
    stopped_epoch: int | None = None
    early_stopping_triggered: bool = False


def collate_dino_views(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    vt = torch.stack([item["view_teacher"] for item in batch], dim=0)
    vs = torch.stack([item["view_student"] for item in batch], dim=0)
    out: dict[str, torch.Tensor] = {"view_teacher": vt, "view_student": vs}
    if "tissue_patch_weights" in batch[0]:
        out["tissue_patch_weights"] = torch.stack([item["tissue_patch_weights"] for item in batch], dim=0)
    return out


class DINOTrainer:
    def __init__(
        self,
        *,
        model: DINOSSLModule,
        center: DINOLogitCenter,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        teacher_momentum: float,
        student_temp: float,
        teacher_temp: float,
        patch_center: DINOLogitCenter | None,
        ibot_weight: float,
        ibot_mask_ratio: float,
        ibot_student_temp: float,
        ibot_teacher_temp: float,
        grad_accum_steps: int,
        amp: bool,
        checkpoints_dir: Path,
        logs_dir: Path,
        ddp: bool = False,
        ddp_rank: int = 0,
        data_parallel: bool = False,
        data_parallel_device_ids: list[int] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.center = center.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.teacher_momentum = float(teacher_momentum)
        self.student_temp = float(student_temp)
        self.teacher_temp = float(teacher_temp)
        self.patch_center = patch_center.to(device) if patch_center is not None else None
        self.ibot_weight = float(ibot_weight)
        self.ibot_mask_ratio = float(ibot_mask_ratio)
        self.ibot_student_temp = float(ibot_student_temp)
        self.ibot_teacher_temp = float(ibot_teacher_temp)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.amp = amp and device.type == "cuda"
        self.checkpoints_dir = checkpoints_dir
        self.logs_dir = logs_dir
        self.ddp = bool(ddp)
        self.ddp_rank = int(ddp_rank)
        self.is_main_process = self.ddp_rank == 0
        self.data_parallel = bool(data_parallel)

        if self.ddp:
            self.model.student_backbone = DDP(
                self.model.student_backbone,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                broadcast_buffers=False,
            )
            self.model.student_head = DDP(
                self.model.student_head,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                broadcast_buffers=False,
            )
            self.model.student_patch_head = DDP(
                self.model.student_patch_head,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                broadcast_buffers=False,
            )
        elif self.data_parallel:
            dp_ids = data_parallel_device_ids or [0]
            self.model.student_backbone = DataParallel(self.model.student_backbone, device_ids=dp_ids)
            self.model.student_head = DataParallel(self.model.student_head, device_ids=dp_ids)
            self.model.student_patch_head = DataParallel(self.model.student_patch_head, device_ids=dp_ids)

        params = (
            list(self.model.student_backbone.parameters())
            + list(self.model.student_head.parameters())
            + list(self.model.student_patch_head.parameters())
        )
        self.optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        if self.is_main_process:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _sample_patch_mask(self, patch_weights: torch.Tensor, num_patches: int) -> torch.Tensor:
        """
        Sample mask indices with optional tissue-aware bias.
        patch_weights: (B,N) non-negative, sum can be arbitrary.
        """
        b, n = patch_weights.shape
        if n != num_patches:
            raise ValueError(f"patch_weights N={n} != num_patches={num_patches}")
        k = max(1, int(round(self.ibot_mask_ratio * num_patches)))
        probs = patch_weights.clamp_min(1e-8)
        probs = probs / probs.sum(dim=1, keepdim=True)
        idx = torch.multinomial(probs, num_samples=k, replacement=False)
        m = torch.zeros((b, num_patches), dtype=torch.bool, device=patch_weights.device)
        m.scatter_(1, idx, True)
        return m

    def _run_epoch(
        self,
        *,
        train: bool,
        max_steps: int | None,
        epoch_idx: int,
        epochs: int,
    ) -> float:
        if train:
            self.model.student_backbone.train()
            self.model.student_head.train()
        else:
            self.model.student_backbone.eval()
            self.model.student_head.eval()

        total_loss = 0.0
        num_batches = 0
        loader = self.train_loader if train else self.val_loader
        self.optimizer.zero_grad(set_to_none=True)

        total_steps = len(loader)
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)

        phase = "train" if train else "val"
        # Match dbt_simclr SimCLRTrainer: default tqdm stream (stdout), single bar per phase.
        pbar = tqdm(
            loader,
            total=total_steps,
            desc=f"epoch {epoch_idx}/{epochs} {phase}",
            leave=False,
            dynamic_ncols=True,
            disable=not self.is_main_process,
        )

        for step_idx, batch in enumerate(pbar):
            if max_steps is not None and step_idx >= max_steps:
                break

            vt = batch["view_teacher"].to(self.device, non_blocking=True)
            vs = batch["view_student"].to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                with torch.amp.autocast("cuda", enabled=self.amp):
                    with torch.no_grad():
                        t_logits = self.model.forward_teacher(vt)
                    s_logits = self.model.forward_student(vs)
                    c = self.center.center
                    dino_loss = dino_cross_entropy(
                        s_logits,
                        t_logits.detach(),
                        c,
                        student_temp=self.student_temp,
                        teacher_temp=self.teacher_temp,
                    )
                    loss = dino_loss
                    if train:
                        center_logits = t_logits.detach()
                        if self.ddp:
                            center_logits = center_logits.clone()
                            dist.all_reduce(center_logits, op=dist.ReduceOp.SUM)
                            center_logits /= dist.get_world_size()
                        self.center.update_from_teacher_logits(center_logits)

                    if self.ibot_weight > 0.0 and self.patch_center is not None:
                        with torch.no_grad():
                            _, t_patch = self.model.forward_teacher_tokens(vt)
                            t_patch_logits = self.model.forward_teacher_patch_logits(t_patch)
                        pweights = batch.get("tissue_patch_weights")
                        if pweights is None:
                            pweights = torch.ones(
                                (vs.size(0), t_patch_logits.size(1)),
                                device=self.device,
                                dtype=torch.float32,
                            )
                        else:
                            pweights = pweights.to(self.device, non_blocking=True).float()
                        patch_mask = self._sample_patch_mask(pweights, num_patches=t_patch_logits.size(1))
                        _, s_patch = self.model.forward_student_masked_tokens(vs, patch_mask)
                        s_patch_logits = self.model.forward_student_patch_logits(s_patch)
                        ibot_loss = masked_patch_cross_entropy(
                            s_patch_logits,
                            t_patch_logits.detach(),
                            patch_mask,
                            self.patch_center.center,
                            student_temp=self.ibot_student_temp,
                            teacher_temp=self.ibot_teacher_temp,
                        )
                        loss = dino_loss + (self.ibot_weight * ibot_loss)
                        if train:
                            masked_teacher_logits = t_patch_logits[patch_mask].detach().reshape(-1, t_patch_logits.shape[-1])
                            if self.ddp and masked_teacher_logits.numel() > 0:
                                masked_teacher_logits = masked_teacher_logits.clone()
                                dist.all_reduce(masked_teacher_logits, op=dist.ReduceOp.SUM)
                                masked_teacher_logits /= dist.get_world_size()
                            self.patch_center.update_from_teacher_logits(masked_teacher_logits)

                if train:
                    loss_to_backprop = loss / self.grad_accum_steps
                    self.scaler.scale(loss_to_backprop).backward()
                    should_step = (step_idx + 1) % self.grad_accum_steps == 0
                    if should_step:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.model.update_teacher(self.teacher_momentum)
                        self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().item())
            num_batches += 1
            if self.is_main_process:
                pbar.set_postfix(loss=f"{(total_loss / max(1, num_batches)):.4f}")

        return total_loss / max(1, num_batches)

    @staticmethod
    def _plot_loss_curves(history: list[dict[str, Any]], out_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train_dino")
        ax.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val_dino")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("DINO Training")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)

    def fit(
        self,
        *,
        epochs: int,
        max_train_steps: int | None = None,
        max_val_steps: int | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
    ) -> DINOTrainOutput:
        history: list[dict[str, Any]] = []
        best_val_loss = float("inf")
        best_epoch = -1
        best_ckpt = self.checkpoints_dir / "best.pt"
        stopped_epoch: int | None = None
        es_triggered = False
        es_patience = early_stopping_patience if early_stopping_patience and early_stopping_patience > 0 else None
        es_wait = 0

        for epoch in range(epochs):
            if hasattr(self.train_loader.dataset, "set_epoch"):
                self.train_loader.dataset.set_epoch(epoch)
            sampler = getattr(self.train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_loss = self._run_epoch(
                train=True,
                max_steps=max_train_steps,
                epoch_idx=epoch + 1,
                epochs=epochs,
            )
            val_loss = self._run_epoch(
                train=False,
                max_steps=max_val_steps,
                epoch_idx=epoch + 1,
                epochs=epochs,
            )
            entry = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            history.append(entry)
            if self.is_main_process:
                print(
                    f"epoch={epoch+1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
                    flush=True,
                )

            improved = val_loss < best_val_loss - float(early_stopping_min_delta)
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                if self.is_main_process:
                    torch.save(
                        {
                            "epoch": best_epoch,
                            "model_state": self.model.state_dict(),
                            "center_state": self.center.state_dict(),
                            "patch_center_state": self.patch_center.state_dict() if self.patch_center is not None else None,
                            "val_loss": best_val_loss,
                        },
                        best_ckpt,
                    )

            if self.is_main_process:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": self.model.state_dict(),
                        "center_state": self.center.state_dict(),
                        "patch_center_state": self.patch_center.state_dict() if self.patch_center is not None else None,
                        "val_loss": val_loss,
                    },
                    self.checkpoints_dir / "last.pt",
                )

            if es_patience is not None:
                if improved:
                    es_wait = 0
                else:
                    es_wait += 1
                if es_wait >= es_patience:
                    stopped_epoch = epoch + 1
                    es_triggered = True
                    if self.is_main_process:
                        print(f"Early stopping at epoch={stopped_epoch}: val_loss plateau.")
                    break

        history_path = self.logs_dir / "history.json"
        if self.is_main_process:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        summary_path = self.logs_dir / "summary.json"
        if self.is_main_process:
            summary_path.write_text(
                json.dumps(
                    {
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "best_checkpoint": str(best_ckpt),
                        "early_stopping": {
                            "enabled": es_patience is not None,
                            "patience": es_patience,
                            "triggered": es_triggered,
                            "stopped_epoch": stopped_epoch,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        plots_dir = self.logs_dir / "plots"
        if self.is_main_process:
            self._plot_loss_curves(history, plots_dir / "training_curves.png")

        if self.ddp:
            dist.barrier()

        return DINOTrainOutput(
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            best_checkpoint=best_ckpt,
            history_path=history_path,
            summary_path=summary_path,
            plots_dir=plots_dir,
            stopped_epoch=stopped_epoch,
            early_stopping_triggered=es_triggered,
        )
