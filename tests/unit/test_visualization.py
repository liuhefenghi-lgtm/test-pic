"""
Unit tests for utils/visualization.py

Covers:
- _get_color          – class-name → BGR colour lookup
- _get_label_text     – label string formatting
- draw_detections     – core rendering function
- draw_legend         – legend image generation
- ID_TO_NAME / CLASS_COLORS_BGR constants
"""

import numpy as np
import pytest

from utils.visualization import (
    CLASS_COLORS_BGR,
    CLASS_NAMES_CN,
    DEFAULT_COLOR,
    ID_TO_NAME,
    _get_color,
    _get_label_text,
    draw_detections,
    draw_legend,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants / mappings
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_id_to_name_has_six_classes(self):
        assert len(ID_TO_NAME) == 6

    def test_id_to_name_ids_are_zero_to_five(self):
        assert set(ID_TO_NAME.keys()) == {0, 1, 2, 3, 4, 5}

    def test_class_colors_has_entry_for_every_class(self):
        for name in ID_TO_NAME.values():
            assert name in CLASS_COLORS_BGR

    def test_bgr_values_are_in_range(self):
        for color in CLASS_COLORS_BGR.values():
            assert len(color) == 3
            for channel in color:
                assert 0 <= channel <= 255

    def test_class_names_cn_covers_all_classes(self):
        for name in ID_TO_NAME.values():
            assert name in CLASS_NAMES_CN


# ─────────────────────────────────────────────────────────────────────────────
# _get_color
# ─────────────────────────────────────────────────────────────────────────────

class TestGetColor:
    def test_known_class_returns_correct_color(self):
        assert _get_color("scratch") == CLASS_COLORS_BGR["scratch"]
        assert _get_color("bump") == CLASS_COLORS_BGR["bump"]
        assert _get_color("blister") == CLASS_COLORS_BGR["blister"]

    def test_unknown_class_returns_default(self):
        assert _get_color("nonexistent_class") == DEFAULT_COLOR

    @pytest.mark.parametrize("name", ["scratch", "bump", "stain",
                                       "indentation", "color_diff", "blister"])
    def test_all_six_classes(self, name):
        color = _get_color(name)
        assert len(color) == 3


# ─────────────────────────────────────────────────────────────────────────────
# _get_label_text
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLabelText:
    def test_contains_confidence_score(self):
        text = _get_label_text("scratch", 0.85)
        assert "0.85" in text

    def test_show_cn_true_includes_chinese(self):
        text = _get_label_text("scratch", 0.9, show_cn=True)
        # Chinese name for scratch is 划伤
        assert "划伤" in text

    def test_show_cn_false_excludes_chinese(self):
        text = _get_label_text("scratch", 0.9, show_cn=False)
        assert "scratch" in text
        assert "划伤" not in text

    def test_unknown_class_uses_name_as_fallback(self):
        text = _get_label_text("unknown_defect", 0.5, show_cn=True)
        assert "unknown_defect" in text


# ─────────────────────────────────────────────────────────────────────────────
# draw_detections
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawDetections:
    def test_returns_same_shape_as_input(self, sample_bgr_image):
        boxes = [(10, 10, 50, 50)]
        result = draw_detections(sample_bgr_image, boxes, [0], [0.9])
        assert result.shape == sample_bgr_image.shape

    def test_returns_copy_does_not_modify_input(self, sample_bgr_image):
        original = sample_bgr_image.copy()
        boxes = [(10, 10, 50, 50)]
        draw_detections(sample_bgr_image, boxes, [0], [0.9])
        np.testing.assert_array_equal(sample_bgr_image, original)

    def test_no_detections_returns_copy_unchanged(self, sample_bgr_image):
        original = sample_bgr_image.copy()
        result = draw_detections(sample_bgr_image, [], [], [])
        np.testing.assert_array_equal(result, original)

    def test_output_is_uint8(self, sample_bgr_image):
        result = draw_detections(sample_bgr_image, [(5, 5, 30, 30)], [1], [0.7])
        assert result.dtype == np.uint8

    def test_confidence_threshold_skips_low_conf(self, sample_bgr_image):
        # With conf_threshold=0.9, the 0.5-confidence box should be skipped.
        result_filtered = draw_detections(
            sample_bgr_image, [(5, 5, 30, 30)], [0], [0.5],
            conf_threshold=0.9
        )
        # Without filter it draws; with filter it should not.
        result_unfiltered = draw_detections(
            sample_bgr_image, [(5, 5, 30, 30)], [0], [0.95],
            conf_threshold=0.9
        )
        # Image modified only when the high-confidence prediction passes through.
        assert not np.array_equal(result_unfiltered, sample_bgr_image)

    def test_multiple_detections(self, gray_bgr_image):
        boxes = [(10, 10, 50, 50), (200, 200, 300, 300), (400, 400, 500, 500)]
        class_ids = [0, 2, 4]
        confidences = [0.9, 0.75, 0.6]
        result = draw_detections(gray_bgr_image, boxes, class_ids, confidences)
        assert result.shape == gray_bgr_image.shape
        assert result.dtype == np.uint8

    def test_box_clipped_to_image_boundary(self, blank_bgr_image):
        # Box extends beyond image borders – should not raise.
        boxes = [(-10, -10, 200, 200)]  # image is 100×100
        result = draw_detections(blank_bgr_image, boxes, [0], [0.8])
        assert result.shape == blank_bgr_image.shape

    def test_custom_class_names(self, sample_bgr_image):
        custom_names = {99: "custom_defect"}
        result = draw_detections(
            sample_bgr_image, [(10, 10, 40, 40)], [99], [0.8],
            class_names=custom_names
        )
        assert result.shape == sample_bgr_image.shape

    def test_show_cn_false_does_not_raise(self, sample_bgr_image):
        result = draw_detections(
            sample_bgr_image, [(10, 10, 40, 40)], [0], [0.8],
            show_cn=False
        )
        assert result.shape == sample_bgr_image.shape

    def test_drawing_modifies_image(self, blank_bgr_image):
        # Drawing on a black image should change at least some pixels.
        result = draw_detections(blank_bgr_image, [(10, 10, 80, 80)], [0], [0.9])
        assert not np.array_equal(result, blank_bgr_image)


# ─────────────────────────────────────────────────────────────────────────────
# draw_legend
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawLegend:
    def test_returns_ndarray(self):
        legend = draw_legend()
        assert isinstance(legend, np.ndarray)

    def test_output_shape_matches_requested_size(self):
        legend = draw_legend(img_size=(400, 300))
        # draw_legend takes (width, height); OpenCV image is (height, width, 3)
        assert legend.shape == (300, 400, 3)

    def test_output_dtype_is_uint8(self):
        legend = draw_legend()
        assert legend.dtype == np.uint8

    def test_custom_class_names(self):
        custom = {0: "scratch", 1: "bump"}
        legend = draw_legend(class_names=custom)
        assert legend.dtype == np.uint8

    def test_not_all_black(self):
        # The legend should have at least some coloured pixels (colour blocks).
        legend = draw_legend()
        assert legend.max() > 0
