"""Predict the disease class of grape-leaf images, with confidence and optional Grad-CAM.

Usage::

    python -m grapevine.predict path/to/leaf.jpg [more.jpg ...]
    python -m grapevine.predict leaf.jpg --gradcam-dir outputs/ --checkpoint artifacts/runs/resnet50/best.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import BEST_MODEL_DIR
from .dataset import eval_transform, load_image
from .gradcam import GradCAM, overlay_heatmap
from .models import gradcam_target_layer, load_checkpoint
from .utils import get_device


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    cam: np.ndarray | None = field(default=None, repr=False)
    overlay: Image.Image | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": self.confidence, "probabilities": self.probabilities}


class Predictor:
    """Loads a checkpoint once and classifies PIL images or image paths."""

    def __init__(self, checkpoint: Path | str = BEST_MODEL_DIR / "model.pt", device: str | None = None):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"{checkpoint} not found - train, evaluate and run `python -m grapevine.compare` first.")
        self.device = get_device(device)
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.model_name: str = ckpt["model_name"]
        self.class_names: list[str] = ckpt["class_names"]
        self.image_size: int = ckpt["image_size"]
        self.checkpoint_info = {k: ckpt[k] for k in ("model_name", "epoch", "val", "splits_sha256")}
        self.transform = eval_transform(self.image_size)
        self.cam = GradCAM(self.model, gradcam_target_layer(self.model, self.model_name))

    def predict(self, image: Image.Image | str | Path, gradcam: bool = False) -> Prediction:
        img = load_image(image) if isinstance(image, (str, Path)) else image.convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        cam = overlay = None
        if gradcam:
            cam, probs, _ = self.cam(x)
            overlay = overlay_heatmap(img, cam)
        else:
            with torch.no_grad():
                probs = torch.softmax(self.model(x).float(), dim=1)[0].cpu().numpy()
        k = int(np.argmax(probs))
        return Prediction(
            label=self.class_names[k],
            confidence=float(probs[k]),
            probabilities={c: float(p) for c, p in zip(self.class_names, probs)},
            cam=cam,
            overlay=overlay,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=BEST_MODEL_DIR / "model.pt")
    parser.add_argument("--gradcam-dir", type=Path, default=None, help="save Grad-CAM overlays here")
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    predictor = Predictor(args.checkpoint, args.device)
    results = []
    for path in args.images:
        pred = predictor.predict(path, gradcam=args.gradcam_dir is not None)
        entry = {"image": str(path), **pred.to_dict()}
        if args.gradcam_dir is not None:
            args.gradcam_dir.mkdir(parents=True, exist_ok=True)
            out = args.gradcam_dir / f"{path.stem}_gradcam.png"
            pred.overlay.save(out)
            entry["gradcam"] = str(out)
        results.append(entry)
        if not args.json:
            probs = ", ".join(f"{c}: {p:.2%}" for c, p in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]))
            print(f"{path.name}: {pred.label} ({pred.confidence:.2%})  [{probs}]"
                  + (f"  Grad-CAM -> {entry['gradcam']}" if "gradcam" in entry else ""))
    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
