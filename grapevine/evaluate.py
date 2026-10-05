"""Evaluate trained checkpoints on the validation and held-out test splits.

Usage::

    python -m grapevine.evaluate --model resnet50
    python -m grapevine.evaluate --all

Metrics are computed on several views of each split, because the raw file counts are
dominated by augmented copies:

* ``all``                 - every image in the split;
* ``original_photos``     - un-augmented JPG photos only (no flip/rotation, no Keras PNG);
* ``one_image_per_leaf``  - one image per source group, so every physical leaf counts once;
* ``png_keras_aug``       - only the Keras-augmented PNG files;
* ``strict_no_weak_train_match`` - drops images with a weak, unlinked same-class SIFT match
  to a training image (possible re-shoot of a training leaf).

Everything written to ``artifacts/reports/<model>/`` is computed from the model's actual
predictions; nothing is estimated or filled in by hand.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .config import MODEL_NAMES, REPORTS_DIR, RUNS_DIR, DedupConfig
from .dataset import LeafDataset, eval_transform, load_splits
from .models import count_parameters, load_checkpoint
from .utils import get_device, project_relative, save_json, sync_device
from . import viz


@torch.no_grad()
def predict_probs(model, frame: pd.DataFrame, image_size: int, device, batch_size: int = 64, num_workers: int = 2) -> np.ndarray:
    loader = DataLoader(LeafDataset(frame, eval_transform(image_size)), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu())
    return torch.cat(out).numpy()


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    y_pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    present = np.unique(y_true)
    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": pm, "macro_recall": rm, "macro_f1": fm,
        "weighted_precision": pw, "weighted_recall": rw, "weighted_f1": fw,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "log_loss": log_loss(y_true, np.clip(probs, 1e-7, 1), labels=labels),
        "errors": int((y_pred != y_true).sum()),
        "per_class": {
            c: {"precision": p[k], "recall": r[k], "f1": f[k], "support": int(s[k])} for k, c in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    # ROC-AUC (one-vs-rest, macro) is only defined when every class is present.
    out["roc_auc_ovr_macro"] = roc_auc_score(y_true, probs, multi_class="ovr", labels=labels) if len(present) == len(labels) else None
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def group_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, n_classes: int,
                       n_boot: int = 1000, seed: int = 0) -> dict:
    """95% intervals for accuracy and macro-F1, resampling whole leaf groups with replacement
    (images of one leaf are correlated, so resampling single images would be too optimistic)."""
    rng = np.random.default_rng(seed)
    codes = pd.factorize(pd.Series(groups))[0]
    members = [np.nonzero(codes == g)[0] for g in range(codes.max() + 1)]
    labels = list(range(n_classes))
    acc, f1 = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        acc.append(float((y_true[idx] == y_pred[idx]).mean()))
        f1.append(float(precision_recall_fscore_support(y_true[idx], y_pred[idx], labels=labels, average="macro",
                                                        zero_division=0)[2]))
    return {"resampled_unit": "leaf group", "n_groups": len(members), "n_boot": n_boot,
            "accuracy_95ci": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
            "macro_f1_95ci": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))]}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    """Exact two-sided binomial interval for an accuracy of k/n (valid even when k == n)."""
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def evaluation_views(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    # One image per source group: its first un-augmented photo if it has one, else its first file.
    order = frame.assign(_rank=(~frame["is_original"]).astype(int)).sort_values(["source_group", "_rank", "path"])
    one_per_leaf = frame.index.isin(order.drop_duplicates("source_group").index)
    return {
        "all": np.ones(len(frame), bool),
        "original_photos": frame["is_original"].to_numpy(bool),
        "one_image_per_leaf": one_per_leaf,
        "png_keras_aug": (frame["source_type"] == "png_keras_aug").to_numpy(),
        # Drops held-out images that have a weak, unlinked SIFT match to some training image
        # (a possible re-shoot of a training leaf) - the most conservative view.
        "strict_no_weak_train_match": (frame["max_inliers_to_train"] < DedupConfig.weak_match_inliers).to_numpy(),
    }


@torch.no_grad()
def black_corner_shortcut_check(model, frame: pd.DataFrame, class_names: list[str], image_size: int, device) -> dict:
    """Only the Healthy folder contains 30-degree rotations with black corners. Re-create that
    artefact on non-Healthy test photos: if predictions drift to Healthy, the model learned it."""
    from torchvision import transforms as T

    from .config import IMAGENET_MEAN, IMAGENET_STD

    sub = frame[frame["is_original"] & (frame["label"] != "Healthy")].reset_index(drop=True)
    healthy = class_names.index("Healthy")
    rotate = T.Compose([T.Resize((image_size, image_size), antialias=True), T.RandomRotation((30, 30), fill=0),
                        T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    out = {"n": int(len(sub))}
    for name, tfm in (("unchanged", eval_transform(image_size)), ("rotated_30deg_black_corners", rotate)):
        loader = DataLoader(LeafDataset(sub, tfm), batch_size=64, num_workers=2)
        pred = torch.cat([model(x.to(device)).argmax(1).cpu() for x, _ in loader]).numpy()
        out[name] = {"accuracy": float((pred == sub["label_idx"].to_numpy()).mean()),
                     "predicted_healthy_rate": float((pred == healthy).mean())}
    return out


@torch.no_grad()
def background_neutralized_check(model, frame: pd.DataFrame, class_names: list[str], image_size: int, device) -> dict:
    """Re-score the test set with the background painted grey. The background alone predicts the
    class far above chance (see background_bias.json), so a large drop here would mean the CNN
    leans on capture conditions rather than on the leaf."""
    from torchvision import transforms as T

    from .config import IMAGENET_MEAN, IMAGENET_STD
    from .dataset import NeutralizeBackground

    tfm = T.Compose([T.Resize((image_size, image_size), antialias=True), NeutralizeBackground(), T.ToTensor(),
                     T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    loader = DataLoader(LeafDataset(frame, tfm), batch_size=64, num_workers=2)
    probs = torch.cat([torch.softmax(model(x.to(device)).float(), 1).cpu() for x, _ in loader]).numpy()
    y = frame["label_idx"].to_numpy()
    out = {}
    for view in ("all", "original_photos"):
        m = evaluation_views(frame)[view]
        full = classification_metrics(y[m], probs[m], class_names)
        out[view] = {k: full[k] for k in ("n", "accuracy", "macro_f1", "errors", "confusion_matrix")}
    return out


@torch.no_grad()
def latency_ms(model, image_size: int, device: torch.device, runs: int = 30) -> float:
    """Median single-image forward time (batch size 1, after warm-up)."""
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(5):
        model(x)
    times = []
    for _ in range(runs):
        sync_device(device)
        t0 = time.perf_counter()
        model(x)
        sync_device(device)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def evaluate_model(name: str, device: torch.device) -> dict:
    ckpt_path = RUNS_DIR / name / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found - train the model first.")
    out_dir = REPORTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_checkpoint(ckpt_path, device)
    class_names = ckpt["class_names"]
    splits = load_splits()

    results = {"model": name, "checkpoint": project_relative(ckpt_path), "best_epoch": ckpt["epoch"], "splits_sha256": ckpt["splits_sha256"]}
    for split in ("val", "test"):
        frame = splits[splits["split"] == split].reset_index(drop=True)
        probs = predict_probs(model, frame, ckpt["image_size"], device)
        y = frame["label_idx"].to_numpy()
        results[split] = {view: classification_metrics(y[m], probs[m], class_names)
                          for view, m in evaluation_views(frame).items() if m.any()}
        if split == "test":
            pred = probs.argmax(1)
            table = frame[["path", "label", "source_type", "aug", "is_original", "source_group"]].copy()
            table["pred"] = [class_names[k] for k in pred]
            table["confidence"] = probs.max(1).round(5)
            table["correct"] = pred == y
            for k, c in enumerate(class_names):
                table[f"prob_{c}"] = probs[:, k].round(5)
            table.to_csv(out_dir / "test_predictions.csv", index=False)
            report_txt = classification_report(y, pred, labels=list(range(len(class_names))), target_names=class_names, digits=4)
            (out_dir / "classification_report_test.txt").write_text(report_txt)
            for view in ("all", "original_photos", "strict_no_weak_train_match"):
                viz.confusion_matrix_plot(np.array(results["test"][view]["confusion_matrix"]), class_names,
                                          out_dir / f"confusion_matrix_test_{view}.png",
                                          f"{name} - test ({view}, n={results['test'][view]['n']})")
            results["black_corner_shortcut_check"] = black_corner_shortcut_check(
                model, frame, class_names, ckpt["image_size"], device)
            results["background_neutralized_test"] = background_neutralized_check(
                model, frame, class_names, ckpt["image_size"], device)
            conf = probs.max(1)
            results["test_confidence"] = {  # top-class probability (label smoothing caps it below 1)
                "median": float(np.median(conf)), "p05": float(np.percentile(conf, 5)), "min": float(conf.min()),
                "median_correct": float(np.median(conf[pred == y])) if (pred == y).any() else None,
                "median_wrong": float(np.median(conf[pred != y])) if (pred != y).any() else None,
            }
            views = evaluation_views(frame)
            leaf = results["test"]["one_image_per_leaf"]
            # Images of one leaf are correlated; one image per leaf gives independent trials.
            results["test_accuracy_exact_95ci_one_image_per_leaf"] = {
                "correct": leaf["n"] - leaf["errors"], "n": leaf["n"],
                "interval": clopper_pearson(leaf["n"] - leaf["errors"], leaf["n"])}
            results["test_bootstrap"] = {
                view: group_bootstrap_ci(y[views[view]], pred[views[view]], frame["source_group"].to_numpy()[views[view]],
                                         len(class_names))
                for view in ("all", "strict_no_weak_train_match")
            }

    results["efficiency"] = {
        "parameters": count_parameters(model)["total"],
        "checkpoint_mb": round(ckpt_path.stat().st_size / 2**20, 2),
        f"latency_ms_batch1_{device.type}": round(latency_ms(model, ckpt["image_size"], device), 2),
        "latency_ms_batch1_cpu": round(latency_ms(model.to("cpu"), ckpt["image_size"], torch.device("cpu")), 2),
    }
    save_json(results, out_dir / "metrics.json")
    t, v = results["test"]["all"], results["val"]["all"]
    print(f"[{name}] val acc {v['accuracy']:.4f} macro-F1 {v['macro_f1']:.4f} | test acc {t['accuracy']:.4f} "
          f"macro-F1 {t['macro_f1']:.4f} (n={t['n']}, errors={t['errors']}) | originals-only test acc "
          f"{results['test']['original_photos']['accuracy']:.4f} -> {out_dir / 'metrics.json'}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=MODEL_NAMES)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = get_device(args.device)
    for name in MODEL_NAMES if args.all else [args.model]:
        evaluate_model(name, device)


if __name__ == "__main__":
    main()
