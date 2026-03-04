"""
Shared pytest fixtures for the PCM defect detection test suite.
"""

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Image fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_bgr_image():
    """A 100x100 black BGR image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def gray_bgr_image():
    """A 640x640 mid-gray BGR image that resembles a metal surface."""
    img = np.full((640, 640, 3), 128, dtype=np.uint8)
    return img


@pytest.fixture
def sample_bgr_image():
    """A small 200x300 BGR image with recognisable colour regions."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[:100, :, :] = [200, 200, 200]  # top half: light gray
    img[100:, :, :] = [60, 60, 60]    # bottom half: dark gray
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def single_box():
    """One detection box [x1, y1, x2, y2] in absolute pixels."""
    return np.array([[10, 20, 50, 60]], dtype=float)


@pytest.fixture
def two_boxes():
    """Two non-overlapping boxes."""
    return np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset directory fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """A temporary directory that is removed after the test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def raw_data_dir(tmp_dir):
    """
    Minimal raw-data directory with 10 images and matching YOLO labels.

    Structure:
        <tmp_dir>/
        ├── images/  img_00.jpg … img_09.jpg
        └── labels/  img_00.txt … img_09.txt
    """
    img_dir = tmp_dir / "images"
    lbl_dir = tmp_dir / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    for i in range(10):
        # Write a tiny JPEG
        img = np.full((64, 64, 3), i * 20, dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i:02d}.jpg"), img)

        # Write a valid YOLO label (class_id cx cy w h)
        with open(lbl_dir / f"img_{i:02d}.txt", "w") as f:
            cls = i % 6
            f.write(f"{cls} 0.50 0.50 0.20 0.20\n")

    return tmp_dir


@pytest.fixture
def split_dataset_dir(tmp_dir):
    """
    YOLO split dataset directory with train/val/test images and labels.

    Structure mirrors what split_dataset() produces.
    """
    for split in ("train", "val", "test"):
        (tmp_dir / "images" / split).mkdir(parents=True)
        (tmp_dir / "labels" / split).mkdir(parents=True)

    counts = {"train": 7, "val": 2, "test": 1}
    for split, n in counts.items():
        for i in range(n):
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_dir / "images" / split / f"{split}_{i}.jpg"), img)
            with open(tmp_dir / "labels" / split / f"{split}_{i}.txt", "w") as f:
                f.write(f"0 0.5 0.5 0.4 0.4\n")

    return tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Detection prediction / ground-truth fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def class_names():
    return {0: "scratch", 1: "bump", 2: "stain"}


@pytest.fixture
def perfect_predictions(class_names):
    """Predictions that exactly match the ground truths (all TP)."""
    ground_truths = [
        {"image_id": "img1", "boxes": [[10, 10, 50, 50]], "class_ids": [0]},
        {"image_id": "img2", "boxes": [[20, 20, 80, 80]], "class_ids": [1]},
    ]
    predictions = [
        {
            "image_id": "img1",
            "boxes": [[10, 10, 50, 50]],
            "class_ids": [0],
            "scores": [0.95],
        },
        {
            "image_id": "img2",
            "boxes": [[20, 20, 80, 80]],
            "class_ids": [1],
            "scores": [0.90],
        },
    ]
    return predictions, ground_truths
