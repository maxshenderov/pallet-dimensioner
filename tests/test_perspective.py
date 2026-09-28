"""Тесты модели камеры-обскуры боковой камеры (vision/perspective.py)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.vision.perspective import (
    camera_nadir_mm,
    clip_below_floor,
    floor_row_px,
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


def base_of_box(centre_x, centre_y, size_x, size_y):
    return np.array([
        [centre_x - size_x / 2, centre_y - size_y / 2], [centre_x + size_x / 2, centre_y - size_y / 2],
        [centre_x + size_x / 2, centre_y + size_y / 2], [centre_x - size_x / 2, centre_y + size_y / 2],
    ])


def silhouette_of_box(centre_x, centre_y, size_x, size_y, height_mm, nadir=(0.0, 0.0)):
    """Силуэт коробки сверху ТАКОЙ, КАКОЙ ЕГО ОТДАЁТ СЕГМЕНТАЦИЯ: одна верхняя грань.

    Геометрически в силуэт попадает и основание, но на стенде до него не доходит: ближняя к
    надиру боковая грань обращена от света, выходит тёмной и в маску не попадает. Замер на
    коробке 122 x 78 x 44, стоящей вертикально: силуэт по короткой оси 62.8 мм при увеличении
    1.409, то есть ровно 44 x 1.409 — верхняя грань без всякого основания.
    """
    m = MOUNT_MM / (MOUNT_MM - height_mm)
    base = base_of_box(centre_x, centre_y, size_x, size_y)
    return ((base - np.asarray(nadir)) * m + np.asarray(nadir)).astype(np.float32)


def silhouette_with_side_faces(centre_x, centre_y, size_x, size_y, height_mm, nadir=(0.0, 0.0)):
    """Силуэт, в который попала и ближняя боковая грань — случай другого освещения."""
    m = MOUNT_MM / (MOUNT_MM - height_mm)
    base = base_of_box(centre_x, centre_y, size_x, size_y)
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
        """Груз далеко от центра — там и вылезала ошибка в четверть по короткой стороне."""
        points = silhouette_of_box(2000.0, 1400.0, 1200.0, 800.0, 1500.0)
        sides = footprint_mm_without_side_faces(points, np.array([0.0, 0.0]), 1500.0, MOUNT_MM)
        assert sides == pytest.approx((1200.0, 800.0))

    def test_visible_side_face_would_be_underestimated(self):
        """Известное ограничение, а не забытый случай.

        Если освещение изменится и ближняя боковая грань начнёт попадать в маску, силуэт
        станет длиннее верхней грани, и деление всего силуэта на увеличение занизит габарит.
        Признак на посту — габарит поехал вниз тем сильнее, чем выше груз.
        """
        points = silhouette_with_side_faces(2000.0, 1400.0, 1200.0, 800.0, 1500.0)
        width, depth = footprint_mm_without_side_faces(
            points, np.array([0.0, 0.0]), 1500.0, MOUNT_MM
        )
        assert width > 1200.0 and depth > 800.0

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


def turned_silhouette(centre_x, centre_y, size_x, size_y, height_mm, degrees, nadir=(0.0, 0.0)):
    """Силуэт коробки, повёрнутой в плоскости пола на `degrees`."""
    base = base_of_box(0.0, 0.0, size_x, size_y)
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    turned = base @ rotation.T + np.array([centre_x, centre_y])
    m = MOUNT_MM / (MOUNT_MM - height_mm)
    return ((turned - np.asarray(nadir)) * m + np.asarray(nadir)).astype(np.float32)


class TestWidthAndDepthLabels:
    """Ширина — вдоль оси x пола, глубина — вдоль y. Не «длинная и короткая».

    Оси пола задаёт таблица меток верхней камеры: на посту 1 линия «метка 2 -> метка 1» идёт
    вдоль x и есть ширина. Боковая камера в определении не участвует.

    Обе прежние версии этого места давали на посту неверные названия: сначала ширина была
    длинной стороной (поворот груза не менял ни одной цифры), потом — стороной, обращённой к
    боковой камере (на посту 1 это ровно обратные названия).
    """

    def test_width_is_the_x_side_even_when_it_is_the_short_one(self):
        """Главный случай: короткая сторона вдоль x всё равно называется шириной."""
        points = silhouette_of_box(2000.0, 1400.0, 800.0, 1200.0, 1500.0)
        width, depth = footprint_mm_without_side_faces(points, np.array([0.0, 0.0]), 1500.0, MOUNT_MM)
        assert (width, depth) == pytest.approx((800.0, 1200.0))

    def test_turning_the_load_swaps_them(self):
        """Груз повернули на 90° — ширина с глубиной обязаны поменяться местами."""
        straight = silhouette_of_box(2000.0, 1400.0, 1200.0, 800.0, 1500.0)
        turned = silhouette_of_box(2000.0, 1400.0, 800.0, 1200.0, 1500.0)

        assert footprint_mm_without_side_faces(
            straight, np.array([0.0, 0.0]), 1500.0, MOUNT_MM) == pytest.approx((1200.0, 800.0))
        assert footprint_mm_without_side_faces(
            turned, np.array([0.0, 0.0]), 1500.0, MOUNT_MM) == pytest.approx((800.0, 1200.0))

    def test_turned_load_keeps_its_own_sides_not_the_bounding_box(self):
        """Повёрнутый груз: стороны — его собственные, названия — по ближайшей оси пола.

        Если бы стороны считались проекцией на оси пола, коробка 1200x800 под 30° дала бы
        описанный прямоугольник 1439x1293 вместо своих сторон.
        """
        points = turned_silhouette(2000.0, 1400.0, 1200.0, 800.0, 1500.0, 30.0)
        assert footprint_mm_without_side_faces(
            points, np.array([0.0, 0.0]), 1500.0, MOUNT_MM) == pytest.approx((1200.0, 800.0), rel=1e-3)

    def test_past_forty_five_degrees_the_labels_change_over(self):
        """Под 60° к оси x ближе уже короткая сторона — она и становится шириной."""
        points = turned_silhouette(2000.0, 1400.0, 1200.0, 800.0, 1500.0, 60.0)
        assert footprint_mm_without_side_faces(
            points, np.array([0.0, 0.0]), 1500.0, MOUNT_MM) == pytest.approx((800.0, 1200.0), rel=1e-3)


class TestFloorRow:
    """Плоскость основания груза: ниже неё в силуэте груза быть не может.

    Через неё убирается то, что маска захватывает СВЯЗНО с грузом и потому не отсекается ни
    выбором контура, ни склейкой кусков: собственная тень, отражение в глянцевом полу, обрывок
    упаковки у ножки. На стенде рулон высотой 100 мм читался как 125 — маска уходила на 156 px
    ниже основания через тень на лежащий рядом скотч.
    """

    def test_row_moves_with_distance(self):
        """Одной константой строку не задать — она уезжает тем ниже, чем ближе груз."""
        near = floor_row_px(540.0, 45600.0, 200.0)
        far = floor_row_px(540.0, 45600.0, 400.0)
        assert near == pytest.approx(768.0)
        assert far == pytest.approx(654.0)
        assert near - far == pytest.approx(114.0)

    def test_uncalibrated_post_changes_nothing(self):
        assert floor_row_px(None, None, 250.0) is None
        contour = np.array([[[10, 20]], [[10, 900]], [[80, 900]]], dtype=np.int32)
        assert np.array_equal(clip_below_floor(contour, None), contour)

    def test_everything_below_the_floor_is_cut(self):
        contour = np.array([[[10, 240]], [[10, 1024]], [[80, 1024]], [[80, 240]]], dtype=np.int32)
        _, y, _, h = cv2.boundingRect(clip_below_floor(contour, 868.0))
        assert (y, y + h - 1) == (240, 868)

    def test_a_silhouette_above_the_floor_is_left_alone(self):
        """Обрезка идёт только вниз: недотянувшийся до основания контур не достраивается,
        иначе занижение высоты стало бы невидимым."""
        contour = np.array([[[10, 240]], [[10, 700]], [[80, 700]], [[80, 240]]], dtype=np.int32)
        assert np.array_equal(clip_below_floor(contour, 868.0), contour)

    def test_the_roll_from_the_bench(self):
        """Живой случай: силуэт 238..1024 при основании на 868."""
        contour = np.array([[[968, 238]], [[968, 1024]], [[1368, 1024]], [[1368, 238]]], dtype=np.int32)
        _, _, _, before = cv2.boundingRect(contour)
        _, _, _, after = cv2.boundingRect(clip_below_floor(contour, 868.0))
        assert before == 787 and after == 631
        assert 125.0 * after / before == pytest.approx(100.2, abs=0.5)
