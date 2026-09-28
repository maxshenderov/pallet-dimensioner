"""Аналитический рендер сцены: проверка геометрии поста без камер и без цеха.

Расстановку камер, требуемую площадь зелёного пола и остаточную ошибку метода можно посчитать
до того, как что-то куплено и покрашено. Сцена собирается из выпуклых тел, силуэт выпуклого
тела — это в точности выпуклая оболочка его спроецированных вершин, поэтому маска получается
точной, без трассировки лучей.

Геометрия перспективы масштабно-инвариантна: всё определяется отношениями. Увеличение верхней
грани m = H / (H − h) при подвесе 4000 и грузе 2000 равно 2.0 — и при подвесе 500 с прототипом
250 оно тоже 2.0. Поэтому одна и та же сцена, уменьшенная в 8 раз, честно моделирует пост на
столе: относительная ошибка переносится один в один, абсолютная — с коэффициентом масштаба.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import cv2
import numpy as np


class Solid(Protocol):
    """Выпуклое тело сцены. Силуэт = выпуклая оболочка проекций vertices()."""

    def vertices(self) -> np.ndarray: ...


@dataclass(frozen=True)
class Box:
    """Прямоугольный груз: паллета, коробка, ярус."""

    centre_xy: tuple[float, float]
    size_xy: tuple[float, float]
    z_from: float
    z_to: float
    yaw_deg: float = 0.0

    def vertices(self) -> np.ndarray:
        half_x, half_y = self.size_xy[0] / 2, self.size_xy[1] / 2
        corners = np.array([[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y]])

        angle = np.deg2rad(self.yaw_deg)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        corners = corners @ rotation.T + np.asarray(self.centre_xy)

        return np.array([[x, y, z] for z in (self.z_from, self.z_to) for x, y in corners], dtype=np.float64)


@dataclass(frozen=True)
class Cylinder:
    """Ролик, стоящий вертикально, — проверочный случай многоуровневого груза."""

    centre_xy: tuple[float, float]
    radius: float
    z_from: float
    z_to: float
    segments: int = 64

    def vertices(self) -> np.ndarray:
        angles = np.linspace(0.0, 2 * np.pi, self.segments, endpoint=False)
        ring = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.radius + np.asarray(self.centre_xy)
        return np.array([[x, y, z] for z in (self.z_from, self.z_to) for x, y in ring], dtype=np.float64)


def pinhole_matrix(image_size: tuple[int, int], horizontal_fov_deg: float) -> np.ndarray:
    """Матрица идеальной камеры по полю зрения — для расчётов до калибровки реального объектива."""
    width, height = image_size
    focal = (width / 2) / np.tan(np.deg2rad(horizontal_fov_deg) / 2)
    return np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]], dtype=np.float64)


def look_at_pose(position_mm: Sequence[float], target_mm: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Поза камеры (rvec, tvec) в соглашении OpenCV: X_cam = R · X_world + t.

    На реальном посту поза берётся из solvePnP по углам красного прямоугольника. Здесь она
    задаётся явно — чтобы перебирать варианты расстановки, которых ещё не существует.
    """
    position = np.asarray(position_mm, dtype=np.float64)
    forward = np.asarray(target_mm, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)

    # Надирная камера смотрит вдоль мировой вертикали, и обычная подсказка «верх — это +Z»
    # вырождается: тогда за верх кадра берём мировую ось Y.
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(forward @ up_hint) > 0.99:
        up_hint = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    rotation = np.stack([right, down, forward])
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec, (-rotation @ position).reshape(3, 1)


def render_silhouette(
    solids: Sequence[Solid],
    image_size: tuple[int, int],
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Маска объекта, какой её увидит камера. Объединение силуэтов тел сцены."""
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    rotation, _ = cv2.Rodrigues(rvec)
    for solid in solids:
        vertices = solid.vertices()
        # Точки позади объектива проецируются в зеркальный мусор — в нашей схеме камеры
        # смотрят на пол сверху, поэтому такое означает ошибку расстановки, а не сцены.
        in_front = (vertices @ rotation.T + tvec.reshape(3))[:, 2] > 0
        if not in_front.all():
            raise ValueError("Тело сцены попало за плоскость объектива — камера расставлена неверно")

        projected, _ = cv2.projectPoints(
            vertices.reshape(-1, 1, 3), rvec, tvec, camera_matrix, np.asarray(dist_coeffs, dtype=np.float64)
        )
        hull = cv2.convexHull(projected.reshape(-1, 2).astype(np.float32))
        cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 255)

    return mask


def pallet_with_roll(scale: float = 1.0) -> list[Solid]:
    """Проверочный случай: ролик ⌀600×2000 на паллете 1200×800×144.

    Габарит истинный — 1200 × 800 × 2144. Одна камера сверху читает его как 1292 × 1292:
    верхняя грань ролика увеличена вдвое с лишним и вылезает за паллету во все стороны.
    """
    pallet_height = 144.0 * scale
    return [
        Box(centre_xy=(0.0, 0.0), size_xy=(1200.0 * scale, 800.0 * scale), z_from=0.0, z_to=pallet_height),
        Cylinder(
            centre_xy=(0.0, 0.0),
            radius=300.0 * scale,
            z_from=pallet_height,
            z_to=pallet_height + 2000.0 * scale,
        ),
    ]


def two_level_load(scale: float = 1.0) -> list[Solid]:
    """Два яруса разной высоты — то, что один датчик высоты меряет неверно принципиально."""
    return [
        Box(centre_xy=(0.0, 0.0), size_xy=(1200.0 * scale, 800.0 * scale), z_from=0.0, z_to=144.0 * scale),
        Box(
            centre_xy=(-250.0 * scale, 0.0),
            size_xy=(600.0 * scale, 700.0 * scale),
            z_from=144.0 * scale,
            z_to=1800.0 * scale,
        ),
    ]
