"""Тесты vision/segmentation.py на синтетических изображениях (без физических камер)."""
import cv2
import numpy as np
import pytest

from src.vision.segmentation import segment_pallet

BACKGROUND_HSV_LOWER = (0, 0, 0)
BACKGROUND_HSV_UPPER = (180, 60, 90)
ROI = (0, 0, 400, 300)


def _frame_with_object(present: bool) -> np.ndarray:
    frame = np.full((300, 400, 3), 40, dtype=np.uint8)  # тёмный фон — попадает в HSV-диапазон фона
    if present:
        cv2.rectangle(frame, (100, 80), (300, 220), (200, 200, 200), thickness=-1)  # светлый объект
    return frame


def test_segment_pallet_finds_object():
    frame = _frame_with_object(present=True)
    contour = segment_pallet(frame, ROI, BACKGROUND_HSV_LOWER, BACKGROUND_HSV_UPPER, min_contour_area_px=1000)
    assert contour is not None
    _, _, w, h = cv2.boundingRect(contour)
    assert w == pytest.approx(200, abs=10)
    assert h == pytest.approx(140, abs=10)


def test_segment_pallet_no_object_returns_none():
    frame = _frame_with_object(present=False)
    contour = segment_pallet(frame, ROI, BACKGROUND_HSV_LOWER, BACKGROUND_HSV_UPPER, min_contour_area_px=1000)
    assert contour is None


def test_segment_pallet_filters_small_noise():
    frame = _frame_with_object(present=True)
    contour = segment_pallet(frame, ROI, BACKGROUND_HSV_LOWER, BACKGROUND_HSV_UPPER, min_contour_area_px=10_000_000)
    assert contour is None


def test_segment_pallet_contour_in_full_frame_coordinates():
    roi = (50, 30, 300, 250)
    frame = np.full((300, 400, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (150, 110), (350, 250), (200, 200, 200), thickness=-1)

    contour = segment_pallet(frame, roi, BACKGROUND_HSV_LOWER, BACKGROUND_HSV_UPPER, min_contour_area_px=1000)
    assert contour is not None
    x, y, w, h = cv2.boundingRect(contour)
    assert x == pytest.approx(150, abs=5)
    assert y == pytest.approx(110, abs=5)
