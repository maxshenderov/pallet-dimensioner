"""Расчёт габаритов паллеты из контуров камер и коррекция перспективы. Чистые функции, без I/O."""
from __future__ import annotations

import cv2
import numpy as np

from ..calibration.aruco_homography import pixels_to_mm


def _rect_sides_mm(contour: np.ndarray, homography: np.ndarray) -> tuple[float, float]:
    rect = cv2.minAreaRect(contour)
    box_px = cv2.boxPoints(rect)
    box_mm = [pixels_to_mm((float(px), float(py)), homography) for px, py in box_px]

    side_a = float(np.hypot(box_mm[0][0] - box_mm[1][0], box_mm[0][1] - box_mm[1][1]))
    side_b = float(np.hypot(box_mm[1][0] - box_mm[2][0], box_mm[1][1] - box_mm[2][1]))
    return max(side_a, side_b), min(side_a, side_b)


def compute_footprint_mm(contour_cam1: np.ndarray, homography_cam1: np.ndarray) -> tuple[float, float]:
    """Bounding box контура камеры 1 (top), переведённый в мм. Возвращает (length_mm, width_mm)."""
    return _rect_sides_mm(contour_cam1, homography_cam1)


def compute_height_and_side_mm(
    contour_cam2: np.ndarray,
    homography_cam2: np.ndarray,
    floor_reference_px: int,
) -> tuple[float, float]:
    """Высота и ширина видимой стороны паллеты по контуру камеры 2 (side). Возвращает (height_mm, side_mm)."""
    x, y, w, h = cv2.boundingRect(contour_cam2)

    top_mm = pixels_to_mm((x + w / 2, y), homography_cam2)
    floor_mm = pixels_to_mm((x + w / 2, floor_reference_px), homography_cam2)
    height_mm = abs(floor_mm[1] - top_mm[1])

    left_mm = pixels_to_mm((x, y + h), homography_cam2)
    right_mm = pixels_to_mm((x + w, y + h), homography_cam2)
    side_mm = abs(right_mm[0] - left_mm[0])

    return float(height_mm), float(side_mm)


def correct_perspective(
    footprint_mm: tuple[float, float],
    measured_height_mm: float,
    mount_height_cam1_mm: float,
    reference_distance_cam1_mm: float,
) -> tuple[float, float]:
    """
    Коррекция footprint камеры 1 на реальную высоту верхней грани паллеты.

    d1 = mount_height_cam1 - measured_height_cam2
    scale_factor = d1 / reference_distance_cam1
    """
    d1 = mount_height_cam1_mm - measured_height_mm
    scale_factor = d1 / reference_distance_cam1_mm
    length_mm, width_mm = footprint_mm
    return length_mm * scale_factor, width_mm * scale_factor


def cross_check(
    corrected_footprint: tuple[float, float],
    side_from_cam2: float,
    tolerance_mm: float,
) -> bool:
    """True, если хотя бы одна из сторон corrected_footprint совпадает с side_from_cam2 в пределах допуска."""
    return any(abs(side - side_from_cam2) <= tolerance_mm for side in corrected_footprint)
