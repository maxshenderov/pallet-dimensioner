"""Тесты vision/dimensions.py на синтетических входных данных (заданные контуры/гомографии)."""
import numpy as np
import pytest

from src.vision.dimensions import (
    compute_footprint_mm,
    compute_height_and_side_mm,
    correct_perspective,
    cross_check,
)

SCALE_HOMOGRAPHY = np.diag([2.0, 2.0, 1.0])  # 1px = 2мм, без поворота
IDENTITY_HOMOGRAPHY = np.eye(3)


def _rect_contour(x1, y1, x2, y2) -> np.ndarray:
    return np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)


def test_compute_footprint_mm_applies_scale():
    contour = _rect_contour(10, 10, 110, 60)  # 100x50 px
    length_mm, width_mm = compute_footprint_mm(contour, SCALE_HOMOGRAPHY)
    assert length_mm == pytest.approx(200.0, abs=1.0)
    assert width_mm == pytest.approx(100.0, abs=1.0)


def test_compute_height_and_side_mm():
    contour = _rect_contour(50, 20, 150, 100)  # верх y=20, x в [50,150]
    height_mm, side_mm = compute_height_and_side_mm(contour, IDENTITY_HOMOGRAPHY, floor_reference_px=150)
    assert height_mm == pytest.approx(130.0, abs=1.0)
    assert side_mm == pytest.approx(100.0, abs=1.0)


def test_correct_perspective_scales_footprint_by_height():
    footprint_mm = (1000.0, 600.0)
    length_mm, width_mm = correct_perspective(
        footprint_mm, measured_height_mm=200.0, mount_height_cam1_mm=3000.0, reference_distance_cam1_mm=3000.0
    )
    expected_scale = (3000.0 - 200.0) / 3000.0
    assert length_mm == pytest.approx(1000.0 * expected_scale)
    assert width_mm == pytest.approx(600.0 * expected_scale)


def test_correct_perspective_zero_height_keeps_footprint_unchanged():
    footprint_mm = (1000.0, 600.0)
    length_mm, width_mm = correct_perspective(
        footprint_mm, measured_height_mm=0.0, mount_height_cam1_mm=3000.0, reference_distance_cam1_mm=3000.0
    )
    assert length_mm == pytest.approx(1000.0)
    assert width_mm == pytest.approx(600.0)


def test_cross_check_passes_within_tolerance():
    assert cross_check((1195.0, 800.0), side_from_cam2=805.0, tolerance_mm=30.0) is True


def test_cross_check_fails_outside_tolerance():
    assert cross_check((1195.0, 800.0), side_from_cam2=900.0, tolerance_mm=30.0) is False
