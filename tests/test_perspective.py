"""Тесты модели камеры-обскуры боковой камеры (vision/perspective.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.vision.perspective import (
    camera_nadir_mm,
    footprint_mm_without_side_faces,
    contour_to_floor_mm,
    focal_length_px,
    focal_length_px_from_markers,
    height_looks_inflated,
    height_mm_from_near_face,
    lens_depth_mm,
    object_distances_mm,
    width_mm_from_pinhole,
)

FOCAL_PX = 4000.0
LENS_HEIGHT_MM = 200.0
PRINCIPAL_Y_PX = 540.0


def project_y_px(height_mm: float, distance_mm: float) -> float:
    """Обратная задача: куда спроецируется точка высотой height_mm на расстоянии distance_mm."""
    return PRINCIPAL_Y_PX + FOCAL_PX * (LENS_HEIGHT_MM - height_mm) / distance_mm


def marker_corners(cx: float, cy: float, half: float = 20.0) -> np.ndarray:
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]], dtype=np.float32)


class TestFocalLength:
    def test_computes_focal_from_known_geometry(self):
        assert focal_length_px(500.0, 250.0, 2000.0) == pytest.approx(4000.0)

    @pytest.mark.parametrize("args", [(0.0, 250.0, 2000.0), (500.0, 0.0, 2000.0), (500.0, 250.0, 0.0)])
    def test_rejects_degenerate_input(self, args):
        assert focal_length_px(*args) is None

    def test_from_markers(self):
        detected = {5: marker_corners(200.0, 600.0), 6: marker_corners(700.0, 600.0)}
        known = {5: (0.0, 0.0), 6: (250.0, 0.0)}
        assert focal_length_px_from_markers(detected, known, 2000.0) == pytest.approx(4000.0)

    def test_from_markers_returns_none_when_marker_missing(self):
        detected = {5: marker_corners(200.0, 600.0)}
        known = {5: (0.0, 0.0), 6: (250.0, 0.0)}
        assert focal_length_px_from_markers(detected, known, 2000.0) is None


class TestSceneGeometry:
    def test_lens_sits_in_front_of_markers(self):
        assert lens_depth_mm(0.0, 1000.0, 1) == pytest.approx(-1000.0)
        assert lens_depth_mm(0.0, 1000.0, -1) == pytest.approx(1000.0)

    def test_contour_to_floor_mm_applies_homography(self):
        homography = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        contour = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        floor_mm = contour_to_floor_mm(contour, homography)
        assert floor_mm.ravel().tolist() == pytest.approx([20.0, 40.0, 60.0, 80.0])

    def test_object_distances_split_near_and_far(self):
        floor_mm = np.array([[0.0, 500.0], [800.0, 1700.0], [400.0, 900.0]])
        near, far = object_distances_mm(floor_mm, lens_depth=-1000.0, depth_axis="y")
        assert (near, far) == pytest.approx((1500.0, 2700.0))

    def test_object_distances_respects_depth_axis(self):
        floor_mm = np.array([[500.0, 0.0], [1700.0, 800.0]])
        near, far = object_distances_mm(floor_mm, lens_depth=-1000.0, depth_axis="x")
        assert (near, far) == pytest.approx((1500.0, 2700.0))


class TestHeight:
    def test_recovers_height_from_near_face(self):
        near = 2000.0
        y_bottom, y_top = project_y_px(0.0, near), project_y_px(1000.0, near)
        height = height_mm_from_near_face(y_top, y_bottom, near, FOCAL_PX)
        assert height == pytest.approx(1000.0)

    def test_same_object_measures_the_same_at_different_distances(self):
        """Суть исправления: масштаб больше не привязан к плоскости калибровочных маркеров."""
        heights = []
        for near in (1000.0, 2000.0, 3500.0):
            y_bottom, y_top = project_y_px(0.0, near), project_y_px(1000.0, near)
            heights.append(height_mm_from_near_face(y_top, y_bottom, near, FOCAL_PX))
        assert heights == pytest.approx([1000.0, 1000.0, 1000.0])

    def test_independent_of_lens_height_and_optical_centre(self):
        """Передняя грань целиком на одном расстоянии, поэтому геометрия камеры сокращается."""
        near = 2000.0
        span_px = FOCAL_PX * 1000.0 / near
        for offset in (0.0, 250.0, -400.0):  # сдвиг оптического центра / наклон камеры
            height = height_mm_from_near_face(offset, offset + span_px, near, FOCAL_PX)
            assert height == pytest.approx(1000.0)

    def test_flags_object_lower_than_the_lens(self):
        assert height_looks_inflated(120.0, LENS_HEIGHT_MM) is True
        assert height_looks_inflated(1500.0, LENS_HEIGHT_MM) is False


class TestWidth:
    def test_visible_face_scales_with_near_distance(self):
        assert width_mm_from_pinhole(800.0, 2000.0, FOCAL_PX) == pytest.approx(400.0)


MOUNT_MM = 4000.0


def silhouette_of_box(centre_x, centre_y, size_x, size_y, height_mm, nadir=(0.0, 0.0)):
    """Силуэт коробки сверху: объединение основания и растянутой верхней грани."""
    m = MOUNT_MM / (MOUNT_MM - height_mm)
    base = np.array([
        [centre_x - size_x / 2, centre_y - size_y / 2], [centre_x + size_x / 2, centre_y - size_y / 2],
        [centre_x + size_x / 2, centre_y + size_y / 2], [centre_x - size_x / 2, centre_y + size_y / 2],
    ])
    top = (base - np.asarray(nadir)) * m + np.asarray(nadir)
    return np.vstack([base, top]).astype(np.float32)


class TestFootprint:
    def test_nadir_is_the_frame_centre_in_mm(self):
        homography = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
        assert camera_nadir_mm(homography, (1920, 1080)).tolist() == pytest.approx([480.0, 270.0])

    def test_object_over_the_nadir(self):
        """Боковых граней не видно — растянута вся верхняя грань, сжимать надо обе стороны."""
        points = silhouette_of_box(0.0, 0.0, 1200.0, 800.0, 1500.0)
        sides = footprint_mm_without_side_faces(points, np.array([0.0, 0.0]), 1500.0, MOUNT_MM)
        assert sides == pytest.approx((1200.0, 800.0))

    def test_object_far_off_centre(self):
        """Видна ближняя к надиру боковая грань: её сторону сжимать нельзя."""
        points = silhouette_of_box(2000.0, 1400.0, 1200.0, 800.0, 1500.0)
        sides = footprint_mm_without_side_faces(points, np.array([0.0, 0.0]), 1500.0, MOUNT_MM)
        assert sides == pytest.approx((1200.0, 800.0))

    def test_same_box_measures_the_same_wherever_it_stands(self):
        measured = [
            side
            for cx, cy in ((0.0, 0.0), (900.0, 0.0), (-1800.0, 700.0), (2500.0, -1300.0))
            for side in footprint_mm_without_side_faces(
                silhouette_of_box(cx, cy, 1200.0, 800.0, 1500.0), np.array([0.0, 0.0]), 1500.0, MOUNT_MM
            )
        ]
        assert measured == pytest.approx([1200.0, 800.0] * 4)

    def test_flat_object_needs_no_correction(self):
        points = silhouette_of_box(2000.0, 0.0, 1200.0, 800.0, 0.0)
        sides = footprint_mm_without_side_faces(points, np.array([0.0, 0.0]), 0.0, MOUNT_MM)
        assert sides == pytest.approx((1200.0, 800.0))
