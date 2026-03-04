"""
Unit tests for utils/metrics.py

Covers:
- compute_iou      – IoU matrix computation
- compute_ap       – Area-under-curve AP
- compute_detection_metrics – Full per-class P/R/F1/AP pipeline
- generate_evaluation_report – JSON report creation
"""

import json
from pathlib import Path

import numpy as np
import pytest

from utils.metrics import (
    compute_ap,
    compute_detection_metrics,
    compute_iou,
    generate_evaluation_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# compute_iou
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeIoU:
    def test_perfect_overlap_returns_one(self):
        box = np.array([[0.0, 0.0, 10.0, 10.0]])
        iou = compute_iou(box, box)
        assert iou.shape == (1, 1)
        assert pytest.approx(iou[0, 0], abs=1e-6) == 1.0

    def test_no_overlap_returns_zero(self):
        box1 = np.array([[0.0, 0.0, 5.0, 5.0]])
        box2 = np.array([[10.0, 10.0, 20.0, 20.0]])
        iou = compute_iou(box1, box2)
        assert pytest.approx(iou[0, 0], abs=1e-6) == 0.0

    def test_half_overlap_known_value(self):
        # box1 [0,0,4,4] area=16; box2 [2,0,6,4] area=16; intersection [2,0,4,4] area=8
        # union = 16+16-8 = 24; IoU = 8/24 = 1/3
        box1 = np.array([[0.0, 0.0, 4.0, 4.0]])
        box2 = np.array([[2.0, 0.0, 6.0, 4.0]])
        iou = compute_iou(box1, box2)
        assert pytest.approx(iou[0, 0], abs=1e-6) == 1.0 / 3.0

    def test_output_shape_n_by_m(self):
        box1 = np.array([[0.0, 0.0, 2.0, 2.0],
                          [5.0, 5.0, 7.0, 7.0]])
        box2 = np.array([[0.0, 0.0, 1.0, 1.0],
                          [3.0, 3.0, 6.0, 6.0],
                          [8.0, 8.0, 9.0, 9.0]])
        iou = compute_iou(box1, box2)
        assert iou.shape == (2, 3)

    def test_iou_values_in_range(self):
        rng = np.random.default_rng(0)
        boxes = rng.uniform(0, 100, (8, 4))
        # Ensure x2 > x1 and y2 > y1
        boxes[:, 2] = boxes[:, 0] + rng.uniform(1, 30, 8)
        boxes[:, 3] = boxes[:, 1] + rng.uniform(1, 30, 8)
        iou = compute_iou(boxes, boxes)
        assert np.all(iou >= 0.0)
        assert np.all(iou <= 1.0 + 1e-6)

    def test_identical_boxes_diagonal_is_one(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0],
                           [5.0, 5.0, 15.0, 15.0]])
        iou = compute_iou(boxes, boxes)
        assert pytest.approx(iou[0, 0], abs=1e-6) == 1.0
        assert pytest.approx(iou[1, 1], abs=1e-6) == 1.0

    def test_contained_box_iou(self):
        # box2 fully inside box1 → IoU = area(box2) / area(box1)
        box1 = np.array([[0.0, 0.0, 10.0, 10.0]])  # area 100
        box2 = np.array([[2.0, 2.0, 5.0, 5.0]])    # area 9
        iou = compute_iou(box1, box2)
        expected = 9.0 / 100.0
        assert pytest.approx(iou[0, 0], abs=1e-6) == expected


# ─────────────────────────────────────────────────────────────────────────────
# compute_ap
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAP:
    def test_perfect_detector_ap_is_one(self):
        recall = np.array([0.0, 0.5, 1.0])
        precision = np.array([1.0, 1.0, 1.0])
        ap = compute_ap(recall, precision)
        assert pytest.approx(ap, abs=1e-4) == 1.0

    def test_zero_precision_ap_is_zero(self):
        recall = np.array([0.0, 0.5, 1.0])
        precision = np.array([0.0, 0.0, 0.0])
        ap = compute_ap(recall, precision)
        assert ap == pytest.approx(0.0, abs=1e-6)

    def test_single_point(self):
        recall = np.array([0.5])
        precision = np.array([0.8])
        ap = compute_ap(recall, precision)
        # Area under the step: from r=0→0.5 prec=0.8, from 0.5→1.0 prec=0.0
        assert 0.0 <= ap <= 1.0

    def test_returns_float(self):
        ap = compute_ap(np.array([0.0, 1.0]), np.array([1.0, 0.5]))
        assert isinstance(ap, float)

    def test_monotone_smoothing(self):
        # Precision fluctuates; AP should still be in [0,1].
        recall = np.linspace(0, 1, 10)
        precision = np.array([1, 0.5, 0.9, 0.4, 0.8, 0.3, 0.7, 0.2, 0.6, 0.1])
        ap = compute_ap(recall, precision)
        assert 0.0 <= ap <= 1.0

    def test_step_curve_known_value(self):
        # Recall jumps 0→0.5→1, precision starts at 1, drops to 0.5
        recall = np.array([0.5, 1.0])
        precision = np.array([1.0, 0.5])
        ap = compute_ap(recall, precision)
        # After monotone smoothing: prec[0]=1.0 (no change), prec[1]=0.5
        # Area: (0.5-0)*1.0 + (1.0-0.5)*0.5 = 0.5 + 0.25 = 0.75
        assert pytest.approx(ap, abs=1e-4) == 0.75


# ─────────────────────────────────────────────────────────────────────────────
# compute_detection_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeDetectionMetrics:
    @pytest.fixture
    def class_names(self):
        return {0: "scratch", 1: "bump"}

    def test_perfect_predictions_high_map(self, class_names):
        gts = [{"image_id": "img1", "boxes": [[0, 0, 50, 50]], "class_ids": [0]}]
        preds = [{"image_id": "img1", "boxes": [[0, 0, 50, 50]],
                  "class_ids": [0], "scores": [0.99]}]
        result = compute_detection_metrics(preds, gts, class_names)
        assert result["mAP50"] == pytest.approx(1.0, abs=1e-3)

    def test_no_predictions_map_is_zero(self, class_names):
        gts = [{"image_id": "img1", "boxes": [[0, 0, 50, 50]], "class_ids": [0]}]
        preds = []
        result = compute_detection_metrics(preds, gts, class_names)
        assert result["mAP50"] == pytest.approx(0.0, abs=1e-6)

    def test_all_false_positives_low_precision(self, class_names):
        # No GT boxes but many predictions → all FP
        gts = [{"image_id": "img1", "boxes": [[0, 0, 10, 10]], "class_ids": [0]}]
        preds = [
            {
                "image_id": "img1",
                "boxes": [[90, 90, 99, 99], [80, 80, 89, 89]],
                "class_ids": [0, 0],
                "scores": [0.9, 0.8],
            }
        ]
        result = compute_detection_metrics(preds, gts, class_names, iou_threshold=0.5)
        scratch = result["per_class"].get("scratch", {})
        assert scratch.get("precision", 1.0) < 0.5

    def test_result_has_required_keys(self, class_names):
        gts = [{"image_id": "a", "boxes": [[0, 0, 5, 5]], "class_ids": [0]}]
        preds = [{"image_id": "a", "boxes": [[0, 0, 5, 5]],
                  "class_ids": [0], "scores": [0.8]}]
        result = compute_detection_metrics(preds, gts, class_names)
        assert "mAP50" in result
        assert "iou_threshold" in result
        assert "per_class" in result

    def test_iou_threshold_affects_tp_fp(self, class_names):
        # Prediction overlaps ~0.4 IoU with GT; should be TP at 0.3 but FP at 0.5
        gt_box = [10, 10, 50, 50]
        pred_box = [20, 10, 60, 50]  # shifted right; IoU ≈ 0.43
        gts = [{"image_id": "i", "boxes": [gt_box], "class_ids": [0]}]
        preds = [{"image_id": "i", "boxes": [pred_box],
                  "class_ids": [0], "scores": [0.9]}]

        result_lo = compute_detection_metrics(preds, gts, class_names, iou_threshold=0.3)
        result_hi = compute_detection_metrics(preds, gts, class_names, iou_threshold=0.5)
        assert result_lo["mAP50"] >= result_hi["mAP50"]

    def test_confidence_threshold_filters_low_conf(self, class_names):
        gts = [{"image_id": "i", "boxes": [[0, 0, 50, 50]], "class_ids": [0]}]
        preds = [{"image_id": "i", "boxes": [[0, 0, 50, 50]],
                  "class_ids": [0], "scores": [0.1]}]
        result = compute_detection_metrics(preds, gts, class_names, conf_threshold=0.5)
        assert result["mAP50"] == pytest.approx(0.0, abs=1e-6)

    def test_duplicate_prediction_only_one_tp(self, class_names):
        """The same GT box should only be matched once (greedy matching)."""
        gts = [{"image_id": "i", "boxes": [[0, 0, 50, 50]], "class_ids": [0]}]
        preds = [
            {
                "image_id": "i",
                "boxes": [[0, 0, 50, 50], [0, 0, 50, 50]],
                "class_ids": [0, 0],
                "scores": [0.9, 0.8],
            }
        ]
        result = compute_detection_metrics(preds, gts, class_names)
        scratch = result["per_class"]["scratch"]
        # 1 TP out of 2 predictions → precision = 0.5
        assert scratch["precision"] <= 0.6

    def test_multiclass_per_class_keys(self, class_names):
        gts = [
            {"image_id": "i1", "boxes": [[0, 0, 10, 10]], "class_ids": [0]},
            {"image_id": "i2", "boxes": [[0, 0, 10, 10]], "class_ids": [1]},
        ]
        preds = [
            {"image_id": "i1", "boxes": [[0, 0, 10, 10]],
             "class_ids": [0], "scores": [0.9]},
            {"image_id": "i2", "boxes": [[0, 0, 10, 10]],
             "class_ids": [1], "scores": [0.9]},
        ]
        result = compute_detection_metrics(preds, gts, class_names)
        assert "scratch" in result["per_class"]
        assert "bump" in result["per_class"]


# ─────────────────────────────────────────────────────────────────────────────
# generate_evaluation_report
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateEvaluationReport:
    def test_creates_json_file(self, tmp_dir):
        metrics = {"overall": {"mAP50": 0.85}, "per_class": {}}
        path = generate_evaluation_report(metrics, str(tmp_dir))
        assert Path(path).exists()

    def test_json_contains_metrics(self, tmp_dir):
        metrics = {"overall": {"mAP50": 0.75}, "per_class": {}}
        path = generate_evaluation_report(metrics, str(tmp_dir))
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        assert report["metrics"]["overall"]["mAP50"] == 0.75

    def test_report_has_timestamp(self, tmp_dir):
        path = generate_evaluation_report({}, str(tmp_dir))
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        assert "report_time" in report

    def test_custom_model_and_dataset_names(self, tmp_dir):
        path = generate_evaluation_report(
            {}, str(tmp_dir), model_name="yolo11n", dataset_name="MyDataset"
        )
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        assert report["model"] == "yolo11n"
        assert report["dataset"] == "MyDataset"

    def test_creates_parent_dirs(self, tmp_dir):
        nested = tmp_dir / "a" / "b" / "c"
        path = generate_evaluation_report({}, str(nested))
        assert Path(path).exists()
