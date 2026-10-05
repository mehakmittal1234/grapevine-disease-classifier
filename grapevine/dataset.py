"""PyTorch datasets, transforms, the group-capped sampler and DataLoader factories."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .config import DATA_ROOT, IMAGENET_MEAN, IMAGENET_STD, SPLITS_CSV
from .utils import seed_worker


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees (lossless, no border artefacts)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img.rotate(90 * k) if k else img


class NeutralizeBackground:
    """Paint the background uniform grey (diagnostic for capture bias, not used in training).

    Background = low-saturation, non-dark pixels (paper/table) plus black rotation fill, kept
    only where connected to the image border so pale parts inside the leaf are left alone.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        from scipy import ndimage

        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
        hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
        candidate = ((hsv[..., 1] < 0.25) & (hsv[..., 2] > 0.2)) | (hsv[..., 2] < 0.08)
        labels, _ = ndimage.label(candidate)
        edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        rgb[np.isin(labels, edge[edge > 0])] = 128
        return Image.fromarray(rgb)


def build_transforms(image_size: int, train: bool) -> T.Compose:
    normalize = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        # JPGs are 256x256 and PNGs 224x224; a plain resize keeps the whole leaf in view.
        return T.Compose([T.Resize((image_size, image_size), antialias=True), *normalize])
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRotate90(),
            # Only the Healthy folder ships 30-degree rotations with black corners. Applying
            # black-filled rotations to every class removes that label shortcut.
            T.RandomApply([T.RandomRotation(30, fill=0)], p=0.25),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            *normalize,
        ]
    )


def eval_transform(image_size: int) -> T.Compose:
    return build_transforms(image_size, train=False)


def load_image(path: Path | str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class LeafDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, data_root: Path = DATA_ROOT):
        self.paths = [data_root / p for p in frame["path"]]
        self.labels = frame["label_idx"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.transform(load_image(self.paths[i])), self.labels[i]


class GroupCappedSampler(Sampler[int]):
    """Each epoch, take every image of small groups but at most ``cap`` images of large ones.

    Every Keras PNG group holds ~56 augmented views of a single leaf. Capping them keeps epochs
    short on a laptop GPU and stops a few leaves from dominating the gradient. A fresh random
    subset is drawn each epoch (seeded by ``seed + epoch``), so all views are used over training.
    """

    def __init__(self, groups: Sequence[str], cap: int, seed: int):
        members: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        self.members = list(members.values())
        self.cap = cap
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return sum(min(len(m), self.cap) for m in self.members)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        chosen: list[int] = []
        for m in self.members:
            chosen.extend(rng.choice(m, self.cap, replace=False).tolist() if len(m) > self.cap else m)
        rng.shuffle(chosen)
        return iter(chosen)


def load_splits(path: Path = SPLITS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m grapevine.prepare_data` first.")
    return pd.read_csv(path)


def make_loader(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    train: bool,
    num_workers: int,
    seed: int,
    group_cap: int = 0,
) -> DataLoader:
    dataset = LeafDataset(frame, build_transforms(image_size, train))
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = False
    if train and group_cap > 0:
        sampler = GroupCappedSampler(frame["source_group"].tolist(), group_cap, seed)
    elif train:
        shuffle = True
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        # Only the training loader keeps its workers alive between epochs; on an 8 GB machine
        # the RAM of idle validation workers matters more than their ~5 s restart per epoch.
        persistent_workers=train and num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
