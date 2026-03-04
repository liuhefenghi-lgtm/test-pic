"""
Unit tests for data/dataset.py

Covers:
- bbox_xyxy_to_yolo      – absolute → normalised conversion
- bbox_yolo_to_xyxy      – normalised → absolute conversion
- Round-trip consistency between the two conversions
- save_yolo_label / load_yolo_label – label file I/O
- split_dataset          – train/val/test partitioning
- verify_dataset         – structure validation & statistics
- create_dataset_yaml    – YAML generation
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from data.dataset import (
    bbox_xyxy_to_yolo,
    bbox_yolo_to_xyxy,
    create_dataset_yaml,
    load_yolo_label,
    save_yolo_label,
    split_dataset,
    verify_dataset,
)


# ─────────────────────────────────────────────────────────────────────────────
# bbox_xyxy_to_yolo
# ─────────────────────────────────────────────────────────────────────────────

class TestBboxXyxyToYolo:
    def test_center_box(self):
        # Full-image box → cx=0.5, cy=0.5, w=1.0, h=1.0
        cx, cy, w, h = bbox_xyxy_to_yolo(0, 0, 100, 100, img_w=100, img_h=100)
        assert pytest.approx(cx) == 0.5
        assert pytest.approx(cy) == 0.5
        assert pytest.approx(w) == 1.0
        assert pytest.approx(h) == 1.0

    def test_quarter_box(self):
        # Top-left quarter of a 200×200 image
        cx, cy, w, h = bbox_xyxy_to_yolo(0, 0, 100, 100, img_w=200, img_h=200)
        assert pytest.approx(cx) == 0.25
        assert pytest.approx(cy) == 0.25
        assert pytest.approx(w) == 0.5
        assert pytest.approx(h) == 0.5

    def test_non_square_image(self):
        cx, cy, w, h = bbox_xyxy_to_yolo(10, 20, 30, 60, img_w=100, img_h=200)
        assert pytest.approx(cx) == 0.20
        assert pytest.approx(cy) == 0.20
        assert pytest.approx(w) == 0.20
        assert pytest.approx(h) == 0.20

    def test_output_values_in_range(self):
        cx, cy, w, h = bbox_xyxy_to_yolo(5, 10, 95, 190, img_w=100, img_h=200)
        for val in (cx, cy, w, h):
            assert 0.0 <= val <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# bbox_yolo_to_xyxy
# ─────────────────────────────────────────────────────────────────────────────

class TestBboxYoloToXyxy:
    def test_center_box_full_image(self):
        x1, y1, x2, y2 = bbox_yolo_to_xyxy(0.5, 0.5, 1.0, 1.0, img_w=100, img_h=100)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 100
        assert y2 == 100

    def test_small_box(self):
        # cx=0.5, cy=0.5, w=0.2, h=0.2 on 100×100
        x1, y1, x2, y2 = bbox_yolo_to_xyxy(0.5, 0.5, 0.2, 0.2, img_w=100, img_h=100)
        assert x1 == 40
        assert y1 == 40
        assert x2 == 60
        assert y2 == 60

    def test_returns_integers(self):
        coords = bbox_yolo_to_xyxy(0.3, 0.4, 0.2, 0.3, img_w=640, img_h=480)
        for c in coords:
            assert isinstance(c, int)


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestBboxRoundTrip:
    @pytest.mark.parametrize("x1,y1,x2,y2,iw,ih", [
        (0,   0,  100, 100, 100, 100),
        (10, 20,   90,  80, 100, 100),
        (0,   0,  320, 240, 640, 480),
        (50, 50,  150, 150, 640, 640),
    ])
    def test_round_trip(self, x1, y1, x2, y2, iw, ih):
        cx, cy, w, h = bbox_xyxy_to_yolo(x1, y1, x2, y2, img_w=iw, img_h=ih)
        rx1, ry1, rx2, ry2 = bbox_yolo_to_xyxy(cx, cy, w, h, img_w=iw, img_h=ih)
        # int() truncation may introduce ±1 pixel error
        assert abs(rx1 - x1) <= 1
        assert abs(ry1 - y1) <= 1
        assert abs(rx2 - x2) <= 1
        assert abs(ry2 - y2) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# save_yolo_label / load_yolo_label
# ─────────────────────────────────────────────────────────────────────────────

class TestYoloLabelIO:
    def test_save_then_load_single_annotation(self, tmp_dir):
        path = str(tmp_dir / "label.txt")
        annotations = [(2, 0.5, 0.5, 0.2, 0.3)]
        save_yolo_label(path, annotations)
        loaded = load_yolo_label(path)
        assert len(loaded) == 1
        cls_id, cx, cy, w, h = loaded[0]
        assert cls_id == 2
        assert pytest.approx(cx, abs=1e-5) == 0.5
        assert pytest.approx(cy, abs=1e-5) == 0.5
        assert pytest.approx(w, abs=1e-5) == 0.2
        assert pytest.approx(h, abs=1e-5) == 0.3

    def test_save_then_load_multiple_annotations(self, tmp_dir):
        path = str(tmp_dir / "multi.txt")
        annotations = [
            (0, 0.1, 0.2, 0.3, 0.4),
            (3, 0.5, 0.5, 0.2, 0.2),
            (5, 0.9, 0.9, 0.1, 0.1),
        ]
        save_yolo_label(path, annotations)
        loaded = load_yolo_label(path)
        assert len(loaded) == 3
        for orig, read in zip(annotations, loaded):
            assert orig[0] == read[0]
            for ov, rv in zip(orig[1:], read[1:]):
                assert pytest.approx(rv, abs=1e-5) == ov

    def test_load_nonexistent_returns_empty(self, tmp_dir):
        result = load_yolo_label(str(tmp_dir / "no_such_file.txt"))
        assert result == []

    def test_save_empty_annotations(self, tmp_dir):
        path = str(tmp_dir / "empty.txt")
        save_yolo_label(path, [])
        loaded = load_yolo_label(path)
        assert loaded == []

    def test_load_ignores_blank_lines(self, tmp_dir):
        path = tmp_dir / "blanks.txt"
        path.write_text("0 0.5 0.5 0.2 0.2\n\n1 0.3 0.3 0.1 0.1\n")
        loaded = load_yolo_label(str(path))
        assert len(loaded) == 2


# ─────────────────────────────────────────────────────────────────────────────
# split_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitDataset:
    def test_split_counts_sum_to_total(self, raw_data_dir, tmp_dir):
        output = tmp_dir / "split_out"
        counts = split_dataset(str(raw_data_dir), str(output), train_ratio=0.7, val_ratio=0.15)
        assert counts["train"] + counts["val"] + counts["test"] == 10

    def test_split_ratios_approximately_correct(self, raw_data_dir, tmp_dir):
        output = tmp_dir / "split_out"
        counts = split_dataset(str(raw_data_dir), str(output), train_ratio=0.7, val_ratio=0.2)
        total = sum(counts.values())
        assert counts["train"] / total >= 0.6
        assert counts["val"] / total >= 0.1

    def test_output_dirs_created(self, raw_data_dir, tmp_dir):
        output = tmp_dir / "split_out"
        split_dataset(str(raw_data_dir), str(output))
        for split in ("train", "val", "test"):
            assert (output / "images" / split).is_dir()
            assert (output / "labels" / split).is_dir()

    def test_files_are_copied(self, raw_data_dir, tmp_dir):
        output = tmp_dir / "split_out"
        split_dataset(str(raw_data_dir), str(output))
        total_images = sum(
            len(list((output / "images" / s).iterdir()))
            for s in ("train", "val", "test")
        )
        assert total_images == 10

    def test_deterministic_with_seed(self, raw_data_dir, tmp_dir):
        out1 = tmp_dir / "run1"
        out2 = tmp_dir / "run2"
        c1 = split_dataset(str(raw_data_dir), str(out1), seed=42)
        c2 = split_dataset(str(raw_data_dir), str(out2), seed=42)
        assert c1 == c2

    def test_different_seeds_may_differ(self, raw_data_dir, tmp_dir):
        out1 = tmp_dir / "s1"
        out2 = tmp_dir / "s2"
        # With only 10 files it is possible (but unlikely) that seeds produce
        # the same partition — so we only check that the function succeeds.
        split_dataset(str(raw_data_dir), str(out1), seed=0)
        split_dataset(str(raw_data_dir), str(out2), seed=99)

    def test_assert_on_invalid_ratios(self, raw_data_dir, tmp_dir):
        with pytest.raises(AssertionError):
            split_dataset(str(raw_data_dir), str(tmp_dir / "bad"),
                          train_ratio=0.6, val_ratio=0.5)  # sum > 1


# ─────────────────────────────────────────────────────────────────────────────
# verify_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyDataset:
    def test_returns_dict_with_splits(self, split_dataset_dir):
        stats = verify_dataset(str(split_dataset_dir))
        assert "train" in stats
        assert "val" in stats
        assert "test" in stats

    def test_image_count_matches(self, split_dataset_dir):
        stats = verify_dataset(str(split_dataset_dir))
        assert stats["train"]["images"] == 7
        assert stats["val"]["images"] == 2
        assert stats["test"]["images"] == 1

    def test_missing_label_detected(self, split_dataset_dir):
        # Remove one label file from train
        label = next((split_dataset_dir / "labels" / "train").iterdir())
        label.unlink()
        stats = verify_dataset(str(split_dataset_dir))
        assert stats["train"]["missing_labels"] >= 1

    def test_invalid_label_detected(self, split_dataset_dir):
        # Overwrite one label with invalid content
        label = next((split_dataset_dir / "labels" / "train").iterdir())
        label.write_text("not valid yolo format line\n")
        stats = verify_dataset(str(split_dataset_dir))
        assert stats["train"]["invalid_labels"] >= 1

    def test_total_key_present(self, split_dataset_dir):
        stats = verify_dataset(str(split_dataset_dir))
        assert "total" in stats
        assert stats["total"]["images"] == 10

    def test_nonexistent_split_skipped(self, split_dataset_dir):
        # Requesting a non-existent split should not raise
        stats = verify_dataset(str(split_dataset_dir), splits=("train", "nonexistent"))
        assert "train" in stats


# ─────────────────────────────────────────────────────────────────────────────
# create_dataset_yaml
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateDatasetYaml:
    def test_creates_yaml_file(self, tmp_dir):
        path = create_dataset_yaml(str(tmp_dir))
        assert Path(path).exists()

    def test_yaml_has_required_keys(self, tmp_dir):
        path = create_dataset_yaml(str(tmp_dir))
        with open(path, encoding="utf-8") as f:
            content = yaml.safe_load(f)
        for key in ("path", "train", "val", "test", "nc", "names"):
            assert key in content

    def test_default_nc_is_six(self, tmp_dir):
        path = create_dataset_yaml(str(tmp_dir))
        with open(path, encoding="utf-8") as f:
            content = yaml.safe_load(f)
        assert content["nc"] == 6

    def test_custom_class_names(self, tmp_dir):
        custom = {0: "cat", 1: "dog"}
        path = create_dataset_yaml(str(tmp_dir), class_names=custom)
        with open(path, encoding="utf-8") as f:
            content = yaml.safe_load(f)
        assert content["nc"] == 2

    def test_custom_output_path(self, tmp_dir):
        out = str(tmp_dir / "my_dataset.yaml")
        path = create_dataset_yaml(str(tmp_dir), output_path=out)
        assert path == out
        assert Path(out).exists()
