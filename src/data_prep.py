"""data_prep.py — Stage 1: discovery, quality validation, versioned splits, transforms. 

Implement: locate the casting folders; run data-quality checks (missing/corrupt/duplicate/
dimension/class-distribution/consistency); build reproducible stratified train/val/test
splits with a versioned snapshot + metadata.json; define preprocessing + augmentation
transforms; and per-image feature extraction used by drift monitoring.
"""
from __future__ import annotations

import hashlib, json, os, random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_data_root(base: Path | None = None) -> Path:
    # TODO 1: search under (base or config.DATA_DIR) for a dir whose train/ has both
    #         ok_front and def_front subfolders; return it.
    # raise NotImplementedError("Locate the casting train/test root")
    input_path = base if base else config.DATA_DIR
    root_path =  input_path / 'casting_data'

    if root_path.exists() and all((root_path / 'train' / folder).exists() and (root_path / 'test'  / folder).exists() for folder in config.CLASSES):
        return root_path
    else:
        raise FileNotFoundError(f"{input_path}: No path found")


def list_images(split_dir: Path) -> list[tuple[Path, int]]:
    items = []
    for cls, idx in config.CLASS_TO_IDX.items():
        for p in sorted((split_dir / cls).glob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, idx))
    return items


def validate_quality(root: Path) -> dict:
    # TODO 1 (data quality): for train+test, count images + class distribution; open each
    #         image (catch corrupt); record non-300x300 dims; md5-hash to find duplicates.
    #         Return a report dict with issues + a 'passed' flag.
    # raise NotImplementedError("Implement data-quality validation")
    report = {
        "total_images": 0, "corrupt_files": [], "duplicates": [],
        "bad_dimensions": [], "class_distribution": {}, "passed": True
    }
    seen_hashes = set()
    for split in ("train", "test"):
        all_images = list_images(root / split)
        report["class_distribution"][split] = {cls: 0 for cls in config.CLASSES}

        for img_path, cls_idx in all_images:
            report["total_images"] += 1
            report["class_distribution"][split][config.IDX_TO_CLASS[cls_idx]] += 1

            # 1.2.1 Corrupt-file & Dimension checks
            try:
                with Image.open(img_path) as im:
                    im.verify() # Corrupt check
                with Image.open(img_path) as im:
                    if im.size != (300, 300):
                        report["bad_dimensions"].append(str(img_path))
            except Exception:
                report["corrupt_files"].append(str(img_path))

            # 1.2.2 Duplicate detention (hashing)
            try:
                with open(img_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in seen_hashes:
                    report["duplicates"].append(str(img_path))
                else:
                    seen_hashes.add(file_hash)
            except Exception:
                # if file unreadable, already flagged as corrupt
                pass

    # 1.2.3 Pass/fail logic
    if report["corrupt_files"] or report["bad_dimensions"] or report["duplicates"]:
        report["passed"] = False

    return report


def build_splits(root: Path, version: str = "v1") -> dict:
    # TODO 1 (versioning): stratified val carve-out from train/; test from test/. Save
    #         {train,val,test}.json file lists + metadata.json (version, date, class defs,
    #         split sizes/distribution, seed) under config.SPLIT_DIR/<version>/.
    # raise NotImplementedError("Build versioned stratified splits + metadata")
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    split_dir = config.SPLIT_DIR / version
    split_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "version": version, "date": datetime.now().isoformat(), "class_defs": config.CLASS_TO_IDX,
        "split_info": {}, "seed": config.RANDOM_SEED
    }

    # Collect train images
    train_images = list_images(root / "train")
    test_images = list_images(root / "test")

    # Separate paths and labels
    X = [str(p) for p, _ in train_images]
    y = [cls for _, cls in train_images]

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED
    )

    # Helper to save file lists
    def save_split(name, X_split, y_split):
        files_list = [{"path": x_path, "label": y_split[ind]} for ind, x_path in enumerate(X_split)]
        out_file = split_dir / f"{name}.json"
        with open(out_file, "w") as f:
            json.dump({"files": files_list}, f, indent=2)
        # record counts
        counts = {
            config.IDX_TO_CLASS[0]: y_split.count(0), config.IDX_TO_CLASS[1]: y_split.count(1)
        }
        report["split_info"][name] = counts

    save_split("train", X_train, y_train)
    save_split("val", X_val, y_val)
    save_split("test", [str(p) for p, _ in test_images], [cls for _, cls in test_images])

    # Metadata snapshot
    meta_file = split_dir / "metadata.json"
    with open(meta_file, "w") as meta:
        json.dump(report, meta, indent=2)

    return report


def load_split(version: str, name: str, root: Path) -> list[tuple[Path, int]]:
    rel = json.loads((config.SPLIT_DIR / version / f"{name}.json").read_text())
    return [(root / r, y) for r, y in rel]


def get_transforms(train: bool):
    from torchvision import transforms
    # TODO 2 (preprocessing + augmentation): Grayscale(3) → Resize(224) → [train: flip,
    #         affine rotation/translate, ColorJitter] → ToTensor → Normalize(ImageNet).
    # raise NotImplementedError("Define the train/eval transforms")

    base_tf = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ]
    if train:
        aug_tf = [
            transforms.RandomHorizontalFlip(p=config.AUG["hflip_p"]),
            transforms.RandomAffine(
                degrees=config.AUG["rotation_degrees"],
                translate=(config.AUG["translate"], config.AUG["translate"])
            ),
            transforms.ColorJitter(
                brightness=config.AUG["brightness"], contrast=config.AUG["contrast"]
            )
        ]

        return transforms.Compose(base_tf + aug_tf)
    else:
        return transforms.Compose(base_tf)


def image_features(img: Image.Image) -> dict:
    # TODO 4 (statistical drift): return brightness, contrast, edge_density, sharpness,
    #         mean_intensity for a PIL image (keys == config.DRIFT_FEATURES).
    # raise NotImplementedError("Extract per-image drift features")

    # Convert to grayscale for feature calculations
    gray = img.convert("L")
    arr = np.array(gray)

    # Brightness: mean pixel value
    brightness = arr.mean()

    # Contrast: standard deviation of pixel value
    contrast = arr.std()

    # Edge density: fraction of edge pixels (using Sobel-like filter)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_density = np.array(edges).mean() / 255.0

    # Sharpness: variance of Laplacian (focus measure)
    laplacian = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = np.var(np.array(laplacian))

    # Mean intensity (same as brightness, but kept separate for drift features)
    mean_intensity = brightness

    return {
        "brightness": float(brightness), "contrast": float(contrast), "edge_density": float(edge_density),
        "sharpness": float(sharpness), "mean_intensity": float(mean_intensity)
    }