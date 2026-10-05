"""Grad-CAM (Selvaraju et al., 2017) heatmaps and the example-grid CLI.

Usage::

    python -m grapevine.gradcam                     # best model, sample of test images
    python -m grapevine.gradcam --model resnet50 --per-class 4

Writes ``artifacts/reports/gradcam/<model>_test_examples.png`` (correct predictions per class)
and ``<model>_test_errors.png`` (misclassified test images, if any).
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .config import BEST_MODEL_DIR, DATA_ROOT, REPORTS_DIR, RUNS_DIR, SEED


class GradCAM:
    """Class-activation map from the gradients flowing into ``target_layer``."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output
        if output.requires_grad:  # skipped during plain no-grad inference
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        """Return (cam in [0, 1] at input resolution, softmax probabilities, explained class)."""
        self.model.eval()
        x = x.detach().clone().requires_grad_(True)  # builds a graph even if weights are frozen
        with torch.enable_grad():
            logits = self.model(x)
            idx = int(logits.argmax(1).item()) if class_idx is None else int(class_idx)
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        acts, grads = self.activations.detach()[0], self.gradients.detach()[0]
        weights = grads.mean(dim=(1, 2))  # global-average-pooled gradients, one per channel
        cam = F.relu((weights[:, None, None] * acts).sum(0))
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = torch.softmax(logits.detach().float(), dim=1)[0]
        return cam.cpu().numpy(), probs.cpu().numpy(), idx

    def remove(self) -> None:
        self._handle.remove()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-coloured CAM over the (resized) RGB image."""
    h, w = cam.shape
    base = np.asarray(image.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return Image.fromarray(np.uint8(np.clip((1 - alpha) * base + alpha * heat, 0, 1) * 255))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="model name (default: the selected best model)")
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--max-errors", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from .predict import Predictor
    from .viz import gradcam_grid

    ckpt = BEST_MODEL_DIR / "model.pt" if args.model is None else RUNS_DIR / args.model / "best.pt"
    predictor = Predictor(ckpt, device=args.device)
    name = predictor.model_name
    preds_csv = REPORTS_DIR / name / "test_predictions.csv"
    if not preds_csv.exists():
        raise FileNotFoundError(f"{preds_csv} not found - run `python -m grapevine.evaluate --model {name}` first.")
    preds = pd.read_csv(preds_csv)
    rng = np.random.default_rng(SEED)

    def render(rows: pd.DataFrame) -> list:
        items = []
        for r in rows.itertuples():
            image = Image.open(DATA_ROOT / r.path).convert("RGB")
            result = predictor.predict(image, gradcam=True)
            caption = f"true {r.label}\npred {result.label} ({result.confidence:.1%})"
            items.append((image.resize((224, 224)), result.overlay, caption))
        return items

    correct = preds[preds["correct"] & preds["is_original"]]
    picks = pd.concat([g.iloc[rng.permutation(len(g))[: args.per_class]] for _, g in correct.groupby("label")])
    out_dir = REPORTS_DIR / "gradcam"
    gradcam_grid(render(picks), out_dir / f"{name}_test_examples.png", cols=args.per_class)
    errors = preds[~preds["correct"]].sort_values("confidence", ascending=False).head(args.max_errors)
    if len(errors):
        gradcam_grid(render(errors), out_dir / f"{name}_test_errors.png", cols=min(4, len(errors)))
    print(f"[{name}] Grad-CAM grids written to {out_dir} ({len(picks)} correct examples, {len(errors)} errors)")


if __name__ == "__main__":
    main()
