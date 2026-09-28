"""Тесты вокселной резки (vision/visual_hull.py) на аналитически отрендеренных сценах.

Здесь зафиксировано то, что показала симуляция при выборе расстановки поста, — чтобы правка
кода не откатила выводы незаметно. В частности: боковые камеры на СОСЕДНИХ сторонах зоны,
а не напротив друг друга (240 мм против 680), и объектив от 70° (при 56° объём не помещается
в кадр).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.vision.synthetic import (
    Box,
    Cylinder,
    look_at_pose,
    pallet_with_roll,
    pinhole_matrix,
    render_silhouette,
    two_level_load,
)
from src.vision.visual_hull import (
    HullBox,
    bounding_box_mm,
    carve,
    floor_area_seen_behind_volume,
    frame_coverage,
    project_grid,
    voxel_grid,
)

IMAGE_SIZE = (640, 360)
# Объектив ВЕРХНИХ камер — узкое место расстановки: с 4 м им надо охватить зону 2000×1500
# и ещё 2200 мм высоты. При 70° в кадр попадает 97 % объёма, при 78 % — 98.9, весь объём
# помещается только с 90°. Боковым камерам хватает и 70°.
FOV_DEG = 90.0
VOXEL_MM = 40.0
# Реальные размеры поста. Больший объём при том же объективе уже не помещается в кадр —
# frame_coverage падает ниже 1.0, и часть вокселей режется просто потому, что её не видно.
ZONE_MM = (2000.0, 1500.0)
VOLUME_MM = 2200.0

# Расстановка реального поста: две камеры под потолком + две сбоку на СОСЕДНИХ сторонах зоны.
POST_CAMERAS = [(-900.0, 0.0, 4000.0), (900.0, 0.0, 4000.0), (4500.0, 0.0, 1200.0), (0.0, 4500.0, 1200.0)]


def measure(scene, cameras=POST_CAMERAS, fov_deg=FOV_DEG) -> tuple[HullBox | None, float]:
    """Полный проход поста: силуэты -> проекция сетки -> резка -> габарит."""
    camera_matrix = pinhole_matrix(IMAGE_SIZE, fov_deg)
    grid = voxel_grid(ZONE_MM, VOLUME_MM, VOXEL_MM)

    masks, projections = [], []
    for position in cameras:
        rvec, tvec = look_at_pose(position, (0.0, 0.0, VOLUME_MM / 2))
        masks.append(render_silhouette(scene, IMAGE_SIZE, rvec, tvec, camera_matrix))
        projections.append(project_grid(grid, rvec, tvec, camera_matrix))

    box = bounding_box_mm(grid, carve(projections, masks), VOXEL_MM)
    coverage = min(frame_coverage(p, IMAGE_SIZE) for p in projections)
    return box, coverage


class TestVoxelGrid:
    def test_covers_the_volume_with_cell_centres(self):
        grid = voxel_grid((200.0, 100.0), 80.0, 20.0)
        assert len(grid) == 10 * 5 * 4
        assert grid[:, 0].min() == pytest.approx(-90.0)
        assert grid[:, 0].max() == pytest.approx(90.0)
        assert grid[:, 2].min() == pytest.approx(10.0)
        assert grid[:, 2].max() == pytest.approx(70.0)

    def test_rejects_nonpositive_voxel(self):
        with pytest.raises(ValueError):
            voxel_grid((200.0, 100.0), 80.0, 0.0)


class TestCarve:
    def test_voxel_survives_only_when_every_camera_sees_it(self):
        masks = [np.zeros((10, 10), np.uint8) for _ in range(2)]
        masks[0][5, 5] = masks[0][5, 6] = 255
        masks[1][5, 5] = 255
        projections = [np.array([[5.0, 5.0], [6.0, 5.0]])] * 2
        assert carve(projections, masks).tolist() == [True, False]

    def test_voxel_outside_the_frame_counts_as_empty(self):
        """Обратное правило («не видели — значит не знаем») оставляет непросмотренные углы
        объёма живыми навсегда, и габарит вырождается в размер измерительного объёма."""
        mask = np.full((10, 10), 255, np.uint8)
        assert carve([np.array([[-5.0, 5.0]])], [mask]).tolist() == [False]

    def test_rejects_mismatched_input(self):
        with pytest.raises(ValueError):
            carve([np.zeros((1, 2))], [])
        with pytest.raises(ValueError):
            carve([], [])


class TestBoundingBox:
    def test_returns_none_when_nothing_survived(self):
        grid = voxel_grid((200.0, 100.0), 80.0, 20.0)
        assert bounding_box_mm(grid, np.zeros(len(grid), bool), 20.0) is None

    def test_measures_from_cell_centres_plus_one_voxel(self):
        grid = np.array([[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [100.0, 50.0, 10.0], [0.0, 50.0, 10.0]])
        box = bounding_box_mm(grid, np.ones(4, bool), 20.0)
        assert (box.height_mm, box.width_mm, box.depth_mm) == pytest.approx((20.0, 120.0, 70.0))


class TestPost:
    """Сцены, на которых выбиралась расстановка. Допуски — с запасом к измеренному."""

    def test_roll_on_pallet(self):
        """Случай, ради которого всё затевалось: одна камера сверху читает глубину 1280 при 800."""
        box, coverage = measure(pallet_with_roll())
        assert coverage == pytest.approx(1.0)
        assert box.width_mm == pytest.approx(1200.0, abs=150.0)
        assert box.depth_mm == pytest.approx(800.0, abs=150.0)
        assert box.height_mm == pytest.approx(2144.0, abs=150.0)

    def test_two_level_load(self):
        """Груз с двумя уровнями — то, что один датчик высоты меряет неверно принципиально."""
        box, _ = measure(two_level_load())
        assert box.width_mm == pytest.approx(1200.0, abs=150.0)
        assert box.depth_mm == pytest.approx(800.0, abs=150.0)
        assert box.height_mm == pytest.approx(1800.0, abs=150.0)

    def test_plain_box(self):
        box, _ = measure([Box((0.0, 0.0), (1200.0, 800.0), 0.0, 1500.0)])
        assert box.height_mm == pytest.approx(1500.0, abs=120.0)

    def test_never_underestimates(self):
        """Резка — внешняя оценка: занизить габарит она не может. Для WMS это и ценно."""
        for scene, truth in (
            (pallet_with_roll(), (2144.0, 1200.0, 800.0)),
            (two_level_load(), (1800.0, 1200.0, 800.0)),
            ([Box((0.0, 0.0), (1200.0, 800.0), 0.0, 1500.0, 30.0)], (1500.0, 1200.0, 800.0)),
            ([Box((0.0, 0.0), (1200.0, 800.0), 0.0, 400.0)], (400.0, 1200.0, 800.0)),
        ):
            box, _ = measure(scene)
            assert box.height_mm >= truth[0] - VOXEL_MM
            assert box.width_mm >= truth[1] - VOXEL_MM
            assert box.depth_mm >= truth[2] - VOXEL_MM

    def test_same_load_measures_the_same_wherever_it_stands(self):
        """Методика заказчика: конфиг не трогаем, двигаем груз — показания обязаны держаться.
        Именно этим тестом на стенде вскрылась привязка масштаба к плоскости маркеров."""
        widths, depths, heights = [], [], []
        for centre in ((0.0, 0.0), (400.0, 250.0), (-400.0, -250.0), (400.0, -250.0), (-400.0, 250.0)):
            box, _ = measure([Box(centre, (800.0, 600.0), 0.0, 1200.0)])
            widths.append(box.width_mm)
            depths.append(box.depth_mm)
            heights.append(box.height_mm)

        # Допуск в четыре вокселя: разброс от дискретизации сетки, а не от места груза в зоне.
        for values in (widths, depths, heights):
            assert max(values) - min(values) <= 4 * VOXEL_MM

    def test_side_cameras_belong_on_adjacent_sides(self):
        """Боковые камеры напротив друг друга не режут две оставшиеся грани: 680 мм против 240."""
        scene = [Box((0.0, 0.0), (1200.0, 800.0), 0.0, 1500.0)]
        adjacent, _ = measure(scene, [(-900.0, 0.0, 4000.0), (900.0, 0.0, 4000.0),
                                      (4500.0, 0.0, 1200.0), (0.0, 4500.0, 1200.0)])
        opposite, _ = measure(scene, [(0.0, -900.0, 4000.0), (0.0, 900.0, 4000.0),
                                      (4500.0, 0.0, 1200.0), (-4500.0, 0.0, 1200.0)])
        assert adjacent.width_mm < opposite.width_mm

    def test_ceiling_only_cameras_are_not_enough(self):
        """План v1 (все камеры под потолком) опровергнут: вертикальных граней с торца не видно."""
        scene = [Box((0.0, 0.0), (1200.0, 800.0), 0.0, 1500.0)]
        ceiling, _ = measure(scene, [(0.0, 0.0, 4000.0), (0.0, -2500.0, 4000.0), (-2500.0, 0.0, 4000.0)])
        post, _ = measure(scene)
        assert post.width_mm < ceiling.width_mm
        assert post.height_mm < ceiling.height_mm

    def test_narrow_lens_does_not_cover_the_volume(self):
        """Ходовой объектив 70° оставляет 3 % объёма вне кадра — при монтаже это надо ловить,
        иначе часть вокселей режется просто потому, что её не видно."""
        _, narrow = measure(pallet_with_roll(), fov_deg=70.0)
        _, wide = measure(pallet_with_roll(), fov_deg=FOV_DEG)
        assert narrow < 1.0
        assert wide == pytest.approx(1.0)


class TestGreenFloorExtent:
    def test_camera_sees_floor_far_behind_the_zone(self):
        """Красить зелёным надо заметно больше размеченного прямоугольника: за грузом косая
        камера видит пол далеко за зоной."""
        area = floor_area_seen_behind_volume(np.array([0.0, -2500.0, 4000.0]), (2000.0, 1500.0), 2200.0)
        assert area[3] == pytest.approx(4722.0, abs=1.0)  # +Y уходит на 4.7 м при зоне до 0.75 м

    def test_never_smaller_than_the_zone_itself(self):
        area = floor_area_seen_behind_volume(np.array([0.0, 0.0, 4000.0]), (2000.0, 1500.0), 2200.0)
        assert area[0] <= -1000.0 and area[1] >= 1000.0
        assert area[2] <= -750.0 and area[3] >= 750.0


class TestSyntheticRenderer:
    def test_cylinder_silhouette_is_narrower_than_the_pallet_from_the_side(self):
        camera_matrix = pinhole_matrix(IMAGE_SIZE, FOV_DEG)
        rvec, tvec = look_at_pose((4500.0, 0.0, 1200.0), (0.0, 0.0, 1200.0))
        roll = render_silhouette([Cylinder((0.0, 0.0), 300.0, 0.0, 2000.0)], IMAGE_SIZE, rvec, tvec, camera_matrix)
        pallet = render_silhouette([Box((0.0, 0.0), (1200.0, 800.0), 0.0, 144.0)], IMAGE_SIZE, rvec, tvec, camera_matrix)
        assert np.count_nonzero(roll.any(axis=0)) < np.count_nonzero(pallet.any(axis=0))

    def test_rejects_scene_behind_the_lens(self):
        camera_matrix = pinhole_matrix(IMAGE_SIZE, FOV_DEG)
        rvec, tvec = look_at_pose((0.0, 0.0, 1000.0), (0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="за плоскость объектива"):
            render_silhouette([Box((0.0, 0.0), (100.0, 100.0), 1500.0, 2000.0)], IMAGE_SIZE, rvec, tvec, camera_matrix)
