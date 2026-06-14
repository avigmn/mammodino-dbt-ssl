#!/usr/bin/env python3
"""Single-GPU DINO-full (+ optional iBOT) training on DBT (additive fork).

Same faithful DINO-full recipe (multi-crop, cosine schedules, teacher-temp
warmup, last-layer freeze, grad clip) PLUS an optional iBOT masked-patch branch
controlled by --w-ibot. Run twice with the SAME settings to get a controlled
ablation:  --w-ibot 0.0  (DINO-only)  vs  --w-ibot 0.1  (DINO+iBOT).

Checkpoint format matches DINO-full ({"student","teacher","center"}) so Liron's
staged head-evaluation (Stage 0 build_backbone_from_checkpoint) reads it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MAMMO = Path("/mnt/data/avi/dino_full_ibot")
DBT = Path("/mnt/md0/Liron/dbt_simclr_project")
for p in (str(MAMMO / "src"), str(DBT / "src"), "/mnt/data/avi/py_packages"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mammodino_ssl.data.dbt_dino_multicrop_dataset import (
    DBTDINOMultiCropDataset, DINOMultiCropConfig, collate_multicrop,
)
from mammodino_ssl.train.dino_multicrop_loss import DINOLoss
from mammodino_ssl.train.schedules import cosine_scheduler
from mammodino_ssl.train.dino_full_trainer import (
    get_params_groups, clip_gradients, cancel_gradients_last_layer,
)

# our additive modules (same dir, copied next to this script on the server)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_full_ibot_module import create_dino_full_ibot
from ibot_patch_loss import iBOTPatchLoss, sample_patch_masks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=MAMMO / "configs/dino_dbt_full.yaml")
    p.add_argument("--data-repo-root", type=Path, default=DBT)
    p.add_argument("--run-name", type=str, default="dino_full_ibot")
    p.add_argument("--output-root", type=Path, default=Path("/mnt/data/avi/dino_full_ibot_runs"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=0, help="0 = use config")
    p.add_argument("--batch-size", type=int, default=0, help="0 = use config")
    p.add_argument("--grad-accum-steps", type=int, default=0)
    p.add_argument("--w-ibot", type=float, default=0.1, help="iBOT loss weight (0 = DINO-only)")
    p.add_argument("--ibot-mask-ratio", type=float, default=0.3)
    p.add_argument("--ibot-tissue-bias", type=float, default=0.6)
    p.add_argument("--patch-out-dim", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--smoke", action="store_true", help="2 train + 1 val steps, 1 epoch")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dcfg, mcfg, tcfg = cfg.get("dataset", {}), cfg.get("model", {}), cfg.get("train", {})

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    epochs = args.epochs or int(tcfg.get("epochs", 100))
    batch_size = args.batch_size or int(tcfg.get("batch_size", 16))
    grad_accum = args.grad_accum_steps or int(tcfg.get("grad_accum_steps", 4))
    image_size = int(mcfg.get("image_size", 224))
    patch_size = int(mcfg.get("patch_size", 16))
    grid = image_size // patch_size
    n_patches = grid * grid

    # --- data ---
    ds_cfg = DINOMultiCropConfig(
        resize_height=int(dcfg.get("resize_height", 256)),
        resize_width=int(dcfg.get("resize_width", 256)),
        normalize=bool(dcfg.get("normalize", True)),
        use_processed_cache=False,
        global_size=image_size,
        local_size=int(dcfg.get("local_size", 96)),
        global_crops_scale_min=float(dcfg.get("global_crops_scale_min", 0.4)),
        global_crops_scale_max=float(dcfg.get("global_crops_scale_max", 1.0)),
        local_crops_scale_min=float(dcfg.get("local_crops_scale_min", 0.05)),
        local_crops_scale_max=float(dcfg.get("local_crops_scale_max", 0.4)),
        local_crops_number=int(dcfg.get("local_crops_number", 6)),
        patch_size=patch_size,
        tissue_percentile=float(dcfg.get("tissue_percentile", 80.0)),
        tissue_crop_bias_prob=float(dcfg.get("tissue_crop_bias_prob", 0.6)),
    )
    paths = {
        "manifest": DBT / "artifacts/manifests/master_manifest.parquet",
        "split": DBT / "artifacts/splits/patient_split_v1.json",
    }
    train_ds = DBTDINOMultiCropDataset(manifest_path=paths["manifest"], split_path=paths["split"], split="train", config=ds_cfg)
    val_ds = DBTDINOMultiCropDataset(manifest_path=paths["manifest"], split_path=paths["split"], split="val", config=ds_cfg)
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), collate_fn=collate_multicrop, drop_last=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"), collate_fn=collate_multicrop)

    # --- model ---
    out_dim = int(mcfg.get("num_prototypes", 4096))
    module = create_dino_full_ibot(
        arch=str(mcfg.get("arch", "tiny")), image_size=image_size, patch_size=patch_size,
        out_dim=out_dim, patch_out_dim=int(args.patch_out_dim), w_ibot=float(args.w_ibot),
        head_hidden_dim=int(mcfg.get("head_hidden_dim", 2048)),
        head_bottleneck_dim=int(mcfg.get("head_bottleneck_dim", 256)),
        head_nlayers=int(mcfg.get("head_nlayers", 3)), norm_last_layer=bool(mcfg.get("norm_last_layer", True)),
    ).to(device)

    n_local = ds_cfg.local_crops_number
    ncrops = 2 + n_local
    dino_loss = DINOLoss(
        out_dim, ncrops,
        warmup_teacher_temp=float(tcfg.get("warmup_teacher_temp", 0.04)),
        teacher_temp=float(tcfg.get("teacher_temp", 0.04)),
        warmup_teacher_temp_epochs=int(tcfg.get("warmup_teacher_temp_epochs", 30)),
        nepochs=epochs, student_temp=float(tcfg.get("student_temp", 0.1)),
        center_momentum=float(tcfg.get("center_momentum", 0.9)),
    ).to(device)
    ibot_loss = None
    if args.w_ibot > 0:
        ibot_loss = iBOTPatchLoss(
            int(args.patch_out_dim),
            warmup_teacher_temp=float(tcfg.get("warmup_teacher_temp", 0.04)),
            teacher_temp=float(tcfg.get("teacher_temp", 0.04)),
            warmup_teacher_temp_epochs=int(tcfg.get("warmup_teacher_temp_epochs", 30)),
            nepochs=epochs, student_temp=float(tcfg.get("student_temp", 0.1)),
            center_momentum=float(tcfg.get("center_momentum", 0.9)),
        ).to(device)

    optimizer = torch.optim.AdamW(get_params_groups(module.student))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    niter = max(1, len(train_loader) // grad_accum)
    lr_sched = cosine_scheduler(float(tcfg.get("lr", 5e-4)), float(tcfg.get("min_lr", 1e-6)), epochs, niter,
                                warmup_epochs=int(tcfg.get("warmup_epochs", 10)))
    wd_sched = cosine_scheduler(float(tcfg.get("weight_decay", 0.04)), float(tcfg.get("weight_decay_end", 0.4)), epochs, niter)
    mom_sched = cosine_scheduler(float(tcfg.get("momentum_teacher", 0.996)), 1.0, epochs, niter)
    clip_grad = float(tcfg.get("clip_grad", 3.0))
    freeze_last = int(tcfg.get("freeze_last_layer", 1))
    es_patience = int((tcfg.get("early_stopping") or {}).get("patience", 5))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"{args.run_name}_{ts}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved.json").write_text(json.dumps({
        "w_ibot": args.w_ibot, "ibot_mask_ratio": args.ibot_mask_ratio, "ibot_tissue_bias": args.ibot_tissue_bias,
        "epochs": epochs, "batch_size": batch_size, "grad_accum": grad_accum, "out_dim": out_dim,
        "patch_out_dim": args.patch_out_dim, "arch": mcfg.get("arch", "tiny"), "image_size": image_size,
    }, indent=2))
    print(f"[ibot] run_dir={run_dir} w_ibot={args.w_ibot} epochs={epochs} bs={batch_size} accum={grad_accum} amp={amp}", flush=True)

    def run_epoch(epoch: int, train: bool, max_steps: int | None):
        loader = train_loader if train else val_loader
        module.student.train(train); module.teacher.train(train)
        tot = td = ti = 0.0; nb = 0; opt_step = 0
        optimizer.zero_grad(set_to_none=True)
        it_base = epoch * niter
        pbar = tqdm(loader, desc=f"{'tr' if train else 'va'} ep{epoch+1}/{epochs}", leave=False, dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            if max_steps is not None and step >= max_steps:
                break
            crops = [c.to(device, non_blocking=True) for c in batch["crops"]]
            n_global = int(batch["n_global_crops"])
            tw = batch.get("tissue_patch_weights")
            if train:
                it = min(it_base + opt_step, len(lr_sched) - 1)
                for i, gp in enumerate(optimizer.param_groups):
                    gp["lr"] = float(lr_sched[it])
                    if i == 0:
                        gp["weight_decay"] = float(wd_sched[it])
            mask = None
            if ibot_loss is not None:
                mask = sample_patch_masks(tw, n_global=n_global, batch=crops[0].shape[0], n_patches=n_patches,
                                          mask_ratio=args.ibot_mask_ratio, tissue_bias=args.ibot_tissue_bias)
                mask = mask.to(device)
            with torch.set_grad_enabled(train), torch.amp.autocast("cuda", enabled=amp):
                with torch.no_grad():
                    t_cls, t_patch = module.teacher(crops[:n_global], want_patch=(ibot_loss is not None))
                s_cls, s_patch = module.student(crops, n_global, mask)
                ld = dino_loss(s_cls, t_cls, epoch, n_global_crops=n_global)
                li = ibot_loss(s_patch, t_patch, mask, epoch) if ibot_loss is not None else torch.zeros((), device=device)
                loss = ld + module.w_ibot * li
            if train:
                should = (step + 1) % grad_accum == 0
                scaler.scale(loss / grad_accum).backward()
                if should:
                    if clip_grad > 0:
                        scaler.unscale_(optimizer); clip_gradients(module.student, clip_grad)
                    cancel_gradients_last_layer(epoch, module.student, freeze_last)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    module.update_teacher(float(mom_sched[min(it, len(mom_sched)-1)])); opt_step += 1
            tot += float(loss.detach()); td += float(ld.detach()); ti += float(li.detach() if ibot_loss is not None else 0.0); nb += 1
            pbar.set_postfix(loss=f"{tot/max(1,nb):.3f}", d=f"{td/max(1,nb):.3f}", i=f"{ti/max(1,nb):.3f}")
        return {"loss": tot/max(1,nb), "dino": td/max(1,nb), "ibot": ti/max(1,nb)}

    history = []; best = float("inf"); best_ep = -1; wait = 0; epochs_run = 0
    ms_tr = 2 if args.smoke else None; ms_va = 1 if args.smoke else None
    n_ep = 1 if args.smoke else epochs
    for epoch in range(n_ep):
        if hasattr(train_ds, "set_epoch"): train_ds.set_epoch(epoch)
        tr = run_epoch(epoch, True, ms_tr)
        with torch.no_grad():
            va = run_epoch(epoch, False, ms_va)
        epochs_run = epoch + 1
        entry = {"epoch": epoch+1, "train_loss": tr["loss"], "train_dino": tr["dino"], "train_ibot": tr["ibot"], "val_loss": va["loss"]}
        history.append(entry)
        print(f"[ibot] epoch={epoch+1} train={tr['loss']:.4f}(dino {tr['dino']:.3f}/ibot {tr['ibot']:.3f}) val={va['loss']:.4f}", flush=True)
        improved = va["loss"] < best
        ck = {"student": module.student.state_dict(), "teacher": module.teacher.state_dict(),
              "center": dino_loss.center.detach().cpu(), "epoch": epoch+1, "val_loss": va["loss"], "w_ibot": args.w_ibot}
        torch.save(ck, run_dir / "checkpoints/last.pt")
        if improved:
            best = va["loss"]; best_ep = epoch+1; wait = 0
            torch.save(ck, run_dir / "checkpoints/best.pt")
        else:
            wait += 1
        (run_dir / "logs/history.json").write_text(json.dumps(history, indent=2))
        if not args.smoke and es_patience > 0 and wait >= es_patience:
            print(f"[ibot] early stop at epoch {epoch+1} (best {best_ep}, val {best:.4f})", flush=True)
            break
    (run_dir / "logs/summary.json").write_text(json.dumps({"best_val_loss": best, "best_epoch": best_ep, "epochs_run": epochs_run, "w_ibot": args.w_ibot}, indent=2))
    print(f"[ibot] DONE best_val={best:.4f} @epoch {best_ep} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
