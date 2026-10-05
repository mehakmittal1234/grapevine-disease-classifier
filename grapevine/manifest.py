"""Scan the class folders and build a per-image manifest.

Two filename conventions occur in this dataset, and both encode which *source leaf* an image
came from. That is what makes a leakage-free split possible:

* ``<uuid>___<tag> <num>[_<aug>][(k)].JPG`` - PlantVillage photos (256x256). Files sharing a
  UUID are the original photo plus offline augmentations (``flipLR``, ``90deg``, ``180deg``,
  ``270deg``, ``new30degFlipLR``).
* ``_<idx>_<rand>[(k)].png`` - written by Keras ``ImageDataGenerator.flow(save_to_dir=...)``
  (224x224). ``idx`` is the index of the source image, so all files of a class that share
  ``idx`` are random augmentations (rotation, zoom, shift, flip) of one leaf.

``(k)`` marks a re-downloaded copy. Byte-identical files are detected by MD5 and only the
first copy is kept.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

JPG_RE = re.compile(
    r"^(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})___"
    r"(?P<tag>.+?) (?P<num>\d+)(?:_(?P<aug>[A-Za-z0-9]+))?(?P<copy>\(\d+\))?\.(?i:jpe?g)$"
)
PNG_RE = re.compile(r"^_(?P<idx>\d+)_(?P<rand>\d+)(?P<copy>\(\d+\))?\.png$", re.IGNORECASE)


def parse_filename(class_name: str, filename: str) -> dict:
    """Return the source type, filename-based group key and augmentation tag of one file."""
    if m := JPG_RE.match(filename):
        aug = m["aug"] or "none"
        return {
            "source_type": "jpg_photo",
            "file_group": f"{class_name}|jpg|{m['uuid']}",
            "aug": aug,
            "is_copy": m["copy"] is not None,
            # Photographer's tag and running photo number: consecutive numbers are often
            # re-shoots of the same leaf, which the near-duplicate search exploits.
            "photo_tag": m["tag"],
            "photo_num": int(m["num"]),
        }
    if m := PNG_RE.match(filename):
        return {
            "source_type": "png_keras_aug",
            "file_group": f"{class_name}|png|{int(m['idx'])}",
            "aug": "keras_random",
            "is_copy": m["copy"] is not None,
            "photo_tag": "",
            "photo_num": -1,
        }
    # Unknown naming: treat the file as its own group (content dedup can still merge it).
    return {
        "source_type": "unknown",
        "file_group": f"{class_name}|file|{filename}",
        "aug": "unknown",
        "is_copy": False,
        "photo_tag": "",
        "photo_num": -1,
    }


def _inspect_image(data: bytes) -> dict:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return {"width": im.width, "height": im.height, "mode": im.mode, "readable": True, "error": ""}
    except Exception as exc:  # corrupt or truncated file
        return {"width": -1, "height": -1, "mode": "", "readable": False, "error": repr(exc)}


def build_manifest(data_root: Path, class_names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for label_idx, class_name in enumerate(class_names):
        folder = data_root / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Class folder not found: {folder}")
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            data = path.read_bytes()
            rows.append(
                {
                    "path": path.relative_to(data_root).as_posix(),
                    "label": class_name,
                    "label_idx": label_idx,
                    "ext": path.suffix.lower().lstrip("."),
                    "bytes": len(data),
                    "md5": hashlib.md5(data).hexdigest(),
                    **parse_filename(class_name, path.name),
                    **_inspect_image(data),
                }
            )
    df = pd.DataFrame(rows)

    # Keep one file per MD5: prefer the non-"(k)" copy, then the alphabetically first path.
    order = df.sort_values(["md5", "is_copy", "path"]).index
    dup_mask = df.loc[order].duplicated("md5", keep="first")
    df["is_exact_duplicate"] = False
    df.loc[dup_mask[dup_mask].index, "is_exact_duplicate"] = True
    df["is_original"] = (df["source_type"] == "jpg_photo") & (df["aug"] == "none") & ~df["is_copy"]
    df["usable"] = df["readable"] & ~df["is_exact_duplicate"]
    return df


def summarize_manifest(df: pd.DataFrame) -> dict:
    usable = df[df["usable"]]
    md5_labels = df.groupby("md5")["label"].nunique()
    return {
        "files_scanned": int(len(df)),
        "unreadable_files": df.loc[~df["readable"], "path"].tolist(),
        "exact_duplicates_removed": int(df["is_exact_duplicate"].sum()),
        "exact_duplicates_across_classes": int((md5_labels > 1).sum()),
        "usable_files": int(len(usable)),
        "files_per_class": usable["label"].value_counts().sort_index().to_dict(),
        "files_per_class_and_type": {
            f"{k[0]} / {k[1]}": int(v) for k, v in usable.groupby(["label", "source_type"]).size().items()
        },
        "image_sizes": {
            f"{k[0]} {k[1]}x{k[2]}": int(v) for k, v in usable.groupby(["ext", "width", "height"]).size().items()
        },
        "augmentation_tags": {
            f"{k[0]} / {k[1]}": int(v) for k, v in usable.groupby(["label", "aug"]).size().items()
        },
        "filename_groups_per_class_and_type": {
            f"{k[0]} / {k[1]}": int(v)
            for k, v in usable.groupby(["label", "source_type"])["file_group"].nunique().items()
        },
        "unparsed_filenames": df.loc[df["source_type"] == "unknown", "path"].tolist(),
    }
