"""Frozen ImageNet MobileNetV2 embeddings, used only to shortlist near-duplicate candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm

from .config import IMAGE_SIZE
from .dataset import eval_transform, load_image


class _ImageOnlyDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = eval_transform(IMAGE_SIZE)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.transform(load_image(self.paths[i]))


@torch.no_grad()
def compute_embeddings(paths: list[Path], device: torch.device, batch_size: int, num_workers: int) -> np.ndarray:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier = nn.Identity()
    model.eval().to(device)
    loader = DataLoader(_ImageOnlyDataset(paths), batch_size=batch_size, num_workers=num_workers)
    chunks = []
    for x in tqdm(loader, desc="embedding", unit="batch"):
        x = x.to(device)
        e = model(x) + model(torch.flip(x, dims=[3]))
        chunks.append(F.normalize(e, dim=1).cpu())
    return torch.cat(chunks).numpy().astype(np.float32)
