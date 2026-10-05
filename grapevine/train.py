"""Train one architecture with two-phase transfer learning on the leakage-free split.

Usage::

    python -m grapevine.train --model mobilenet_v2
    python -m grapevine.train --model resnet50 --finetune-epochs 6
    python -m grapevine.train --model efficientnet_b0 --max-batches 5   # quick smoke test

Phase 1 trains only the new head on the frozen ImageNet backbone; phase 2 fine-tunes the
deeper stages with a warm-up + cosine learning-rate schedule. The checkpoint with the best
*validation* macro-F1 is kept (ties broken by lower validation loss), and phase 2 stops early
after ``patience`` epochs without improvement. The test split is never touched here.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, RUNS_DIR, SPLITS_CSV, train_config_for
from .dataset import load_splits, make_loader
from .models import build_model, configure_phase, count_parameters, set_train_mode
from .utils import environment_info, get_device, save_json, set_seed, sync_device


def run_epoch(model, loader, criterion, device, *, optimizer=None, scheduler=None, scaler=None,
              amp=False, frozen=(), max_batches=0) -> dict:
    """One pass over ``loader``; trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    if training:
        set_train_mode(model, list(frozen))
    else:
        model.eval()
    loss_sum, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for step, (x, y) in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            loss_sum += loss.item() * len(y)
            preds.append(logits.argmax(1).cpu())
            labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {
        "loss": loss_sum / len(labels),
        "acc": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=list(range(len(CLASS_NAMES))), zero_division=0)),
        "n": int(len(labels)),
    }


def warmup_cosine(total_steps: int, warmup_steps: int, floor: float = 0.01):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return factor


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(cfg, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    device = get_device()
    amp = cfg.amp and device.type in ("mps", "cuda")
    run_dir = RUNS_DIR / (run_name or cfg.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json({**cfg.to_dict(), "amp_effective": amp, "device": str(device)}, run_dir / "config.json")

    splits = load_splits()
    train_df, val_df = splits[splits["split"] == "train"], splits[splits["split"] == "val"]
    train_loader = make_loader(train_df, cfg.image_size, cfg.batch_size, True, cfg.num_workers, cfg.seed, cfg.group_cap)
    val_loader = make_loader(val_df, cfg.image_size, cfg.batch_size, False, cfg.num_workers, cfg.seed)
    print(f"[{cfg.model}] device={device} amp={amp} train images/epoch={len(train_loader.sampler)} "
          f"(of {len(train_df)}) val={len(val_df)}", flush=True)

    model = build_model(cfg.model, len(CLASS_NAMES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler(device.type) if amp else None

    history, best = [], {"macro_f1": -1.0, "loss": float("inf"), "epoch": None}
    epoch, stale, t_train = 0, 0, time.time()
    phases = [("head", cfg.head_epochs, cfg.head_lr), ("finetune", cfg.finetune_epochs, cfg.finetune_lr)]
    for phase, n_epochs, lr in phases:
        if n_epochs <= 0:
            continue
        frozen = configure_phase(model, cfg.model, phase)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
        steps_per_epoch = min(len(train_loader), cfg.max_batches) if cfg.max_batches else len(train_loader)
        scheduler = None
        if phase == "finetune":
            total = n_epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total, max(1, steps_per_epoch // 2)))
        counts = count_parameters(model)
        print(f"[{cfg.model}] phase={phase} lr={lr} trainable params={counts['trainable']:,}/{counts['total']:,}", flush=True)
        stale = 0
        for _ in range(n_epochs):
            epoch += 1
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            t0 = time.time()
            tr = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, amp=amp, frozen=frozen, max_batches=cfg.max_batches)
            va = run_epoch(model, val_loader, criterion, device, amp=amp, max_batches=cfg.max_batches)
            sync_device(device)
            row = {"epoch": epoch, "phase": phase, "lr_end": optimizer.param_groups[0]["lr"],
                   "train_loss": tr["loss"], "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["acc"],
                   "val_macro_f1": va["macro_f1"], "seconds": round(time.time() - t0, 1)}
            history.append(row)
            improved = va["macro_f1"] > best["macro_f1"] + 1e-6 or (
                abs(va["macro_f1"] - best["macro_f1"]) <= 1e-6 and va["loss"] < best["loss"])
            if improved:
                best = {"macro_f1": va["macro_f1"], "loss": va["loss"], "acc": va["acc"], "epoch": epoch, "phase": phase}
                torch.save({
                    "model_name": cfg.model, "class_names": list(CLASS_NAMES), "image_size": cfg.image_size,
                    "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "state_dict": model.state_dict(),
                    "epoch": epoch, "phase": phase, "val": best, "train_config": cfg.to_dict(),
                    "splits_sha256": file_sha256(SPLITS_CSV), "env": environment_info(),
                }, run_dir / "best.pt")
                stale = 0
            else:
                stale += 1
            print(f"[{cfg.model}] epoch {epoch:2d} {phase:8s} train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                  f"val loss {va['loss']:.4f} acc {va['acc']:.4f} macro-F1 {va['macro_f1']:.4f} | "
                  f"{row['seconds']:.0f}s{' *best*' if improved else ''}", flush=True)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            if phase == "finetune" and stale >= cfg.patience:
                print(f"[{cfg.model}] early stopping: no val macro-F1 gain for {cfg.patience} epochs", flush=True)
                break
        if device.type == "mps":
            torch.mps.empty_cache()

    summary = {
        "model": cfg.model,
        "best_epoch": best["epoch"],
        "best_phase": best.get("phase"),
        "best_val_macro_f1": best["macro_f1"],
        "best_val_acc": best.get("acc"),
        "best_val_loss": best["loss"],
        "epochs_run": epoch,
        "train_seconds": round(time.time() - t_train, 1),
        "parameters": count_parameters(model)["total"],
        "device": str(device),
        "amp": amp,
    }
    save_json(summary, run_dir / "train_summary.json")
    from .viz import training_curves

    training_curves(pd.DataFrame(history), run_dir / "training_curves.png", f"{cfg.model} training (best epoch {best['epoch']})")
    print(f"[{cfg.model}] done in {summary['train_seconds'] / 60:.1f} min; best val macro-F1 "
          f"{best['macro_f1']:.4f} at epoch {best['epoch']} -> {run_dir / 'best.pt'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    for name, typ in [("batch_size", int), ("head_epochs", int), ("finetune_epochs", int), ("head_lr", float),
                      ("finetune_lr", float), ("patience", int), ("group_cap", int), ("num_workers", int),
                      ("seed", int), ("max_batches", int), ("image_size", int)]:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--run-name", default=None, help="output folder under artifacts/runs (default: model name)")
    args = vars(parser.parse_args())
    run_name = args.pop("run_name")
    cfg = train_config_for(args.pop("model"), **args)
    np.set_printoptions(precision=4)
    train(cfg, run_name)


if __name__ == "__main__":
    main()
