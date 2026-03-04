"""
Unit tests for data/augmentation.py

Covers:
- apply_metal_specific_augmentation – output shape, dtype, value range
- get_train_transforms / get_val_transforms – pipeline construction and
  application (skipped if albumentations is not installed)
"""

import numpy as np
import pytest

from data.augmentation import (
    HAS_ALBUMENTATIONS,
    apply_metal_specific_augmentation,
    get_train_transforms,
    get_val_transforms,
)

# Mark all transform tests to skip when albumentations is absent.
requires_albumentations = pytest.mark.skipif(
    not HAS_ALBUMENTATIONS,
    reason="albumentations not installed",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_metal_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Return a synthetic metal-surface BGR image."""
    img = np.full((h, w, 3), 160, dtype=np.uint8)
    # Horizontal stripe pattern typical of rolled metal
    img[::20, :] = 120
    return img


def _make_yolo_boxes(n: int = 2):
    """Return n random valid YOLO bboxes (cx, cy, w, h) and class labels."""
    rng = np.random.default_rng(0)
    boxes, labels = [], []
    for _ in range(n):
        cx = rng.uniform(0.2, 0.8)
        cy = rng.uniform(0.2, 0.8)
        bw = rng.uniform(0.05, 0.2)
        bh = rng.uniform(0.05, 0.2)
        boxes.append([cx, cy, bw, bh])
        labels.append(0)
    return boxes, labels


# ─────────────────────────────────────────────────────────────────────────────
# apply_metal_specific_augmentation
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyMetalSpecificAugmentation:
    def test_output_shape_matches_input(self):
        img = _make_metal_image(128, 128)
        out = apply_metal_specific_augmentation(img)
        assert out.shape == img.shape

    def test_output_dtype_is_uint8(self):
        img = _make_metal_image()
        out = apply_metal_specific_augmentation(img)
        assert out.dtype == np.uint8

    def test_pixel_values_in_valid_range(self):
        img = _make_metal_image()
        out = apply_metal_specific_augmentation(img)
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_non_square_image(self):
        img = _make_metal_image(h=480, w=640)
        out = apply_metal_specific_augmentation(img)
        assert out.shape == (480, 640, 3)

    def test_does_not_modify_input(self):
        img = _make_metal_image()
        original = img.copy()
        apply_metal_specific_augmentation(img)
        np.testing.assert_array_equal(img, original)

    def test_deterministic_with_fixed_seed(self):
        img = _make_metal_image()
        np.random.seed(42)
        out1 = apply_metal_specific_augmentation(img)
        np.random.seed(42)
        out2 = apply_metal_specific_augmentation(img)
        np.testing.assert_array_equal(out1, out2)

    def test_all_black_image(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = apply_metal_specific_augmentation(img)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_all_white_image(self):
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        out = apply_metal_specific_augmentation(img)
        assert int(out.max()) <= 255


# ─────────────────────────────────────────────────────────────────────────────
# get_train_transforms
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTrainTransforms:
    @requires_albumentations
    def test_returns_compose_object(self):
        import albumentations as A
        transform = get_train_transforms(img_size=320)
        assert isinstance(transform, A.Compose)

    @requires_albumentations
    def test_output_image_has_correct_size(self):
        transform = get_train_transforms(img_size=320)
        img = _make_metal_image(480, 640)
        boxes, labels = _make_yolo_boxes(2)
        result = transform(image=img, bboxes=boxes, class_labels=labels)
        out_img = result["image"]
        assert out_img.shape == (320, 320, 3)

    @requires_albumentations
    def test_output_dtype_is_uint8(self):
        transform = get_train_transforms(img_size=320)
        img = _make_metal_image(256, 256)
        boxes, labels = _make_yolo_boxes(1)
        result = transform(image=img, bboxes=boxes, class_labels=labels)
        assert result["image"].dtype == np.uint8

    @requires_albumentations
    def test_bboxes_remain_normalised(self):
        transform = get_train_transforms(img_size=320)
        img = _make_metal_image(256, 256)
        boxes, labels = _make_yolo_boxes(3)
        result = transform(image=img, bboxes=boxes, class_labels=labels)
        for box in result["bboxes"]:
            cx, cy, w, h = box
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0

    @requires_albumentations
    def test_raises_without_albumentations(self, monkeypatch):
        import data.augmentation as aug_module
        monkeypatch.setattr(aug_module, "HAS_ALBUMENTATIONS", False)
        with pytest.raises(ImportError):
            aug_module.get_train_transforms()

    @requires_albumentations
    def test_custom_img_size(self):
        import albumentations as A
        transform = get_train_transforms(img_size=416)
        assert isinstance(transform, A.Compose)


# ─────────────────────────────────────────────────────────────────────────────
# get_val_transforms
# ─────────────────────────────────────────────────────────────────────────────

class TestGetValTransforms:
    @requires_albumentations
    def test_returns_compose_object(self):
        import albumentations as A
        transform = get_val_transforms(img_size=320)
        assert isinstance(transform, A.Compose)

    @requires_albumentations
    def test_output_image_has_correct_size(self):
        transform = get_val_transforms(img_size=320)
        img = _make_metal_image(480, 640)
        boxes, labels = _make_yolo_boxes(2)
        result = transform(image=img, bboxes=boxes, class_labels=labels)
        assert result["image"].shape == (320, 320, 3)

    @requires_albumentations
    def test_output_dtype_is_uint8(self):
        transform = get_val_transforms(img_size=320)
        img = _make_metal_image()
        boxes, labels = _make_yolo_boxes(1)
        result = transform(image=img, bboxes=boxes, class_labels=labels)
        assert result["image"].dtype == np.uint8

    @requires_albumentations
    def test_val_transform_is_deterministic(self):
        """Validation transforms should produce identical outputs on repeated calls."""
        transform = get_val_transforms(img_size=320)
        img = _make_metal_image(300, 300)
        boxes, labels = _make_yolo_boxes(1)
        r1 = transform(image=img.copy(), bboxes=boxes, class_labels=labels)
        r2 = transform(image=img.copy(), bboxes=boxes, class_labels=labels)
        np.testing.assert_array_equal(r1["image"], r2["image"])

    @requires_albumentations
    def test_raises_without_albumentations(self, monkeypatch):
        import data.augmentation as aug_module
        monkeypatch.setattr(aug_module, "HAS_ALBUMENTATIONS", False)
        with pytest.raises(ImportError):
            aug_module.get_val_transforms()
