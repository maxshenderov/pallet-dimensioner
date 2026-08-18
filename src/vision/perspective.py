"""Модель камеры-обскуры для боковой камеры: масштаб, пересчитываемый под каждый объект.

Плоская гомография боковой камеры (два маркера -> подобие) даёт ОДИН фиксированный масштаб
«мм на пиксель», снятый с калибровочных маркеров. Объект, стоящий ближе маркеров, попадает
в кадр крупнее — и его высота завышается пропорционально разнице расстояний. Для паллет это
неустранимо подбором места: паллеты разной глубины, их передние грани всегда на разном
расстоянии от камеры.

Расстояние до объекта здесь берётся из ВЕРХНЕЙ камеры: её гомография построена по четырём
маркерам в плоскости пола, поэтому она даёт не только размер, но и координаты объекта в мм.
Зная расстояние D до передней грани и фокусное расстояние камеры в пикселях:

    height_mm = height_px * D / focal_px

Фокусное расстояние не хранится в конфиге, а считается на каждом кадре по тем же маркерам
5/6 — нужен только один замер рулеткой при установке поста (объектив -> маркеры).
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

DepthAxis = Literal["x", "y"]


def focal_length_px(
    marker_pixel_distance_px: float,
    marker_real_distance_mm: float,
    lens_to_markers_mm: float,
) -> float | None:
    """Фокусное расстояние в пикселях по паре маркеров, снятых с известного расстояния."""
    if marker_pixel_distance_px <= 0 or marker_real_distance_mm <= 0 or lens_to_markers_mm <= 0:
        return None
    return marker_pixel_distance_px * lens_to_markers_mm / marker_real_distance_mm


def focal_length_px_from_markers(
    detected_markers: dict[int, np.ndarray],
    known_positions_mm: dict[int, tuple[float, float]],
    lens_to_markers_mm: float,
) -> float | None:
    """Фокусное по двум калибровочным маркерам боковой камеры. None, если пары нет в кадре."""
    ids = sorted(known_positions_mm)
    if len(ids) != 2 or not set(ids).issubset(detected_markers):
        return None

    (ax, ay), (bx, by) = (known_positions_mm[i] for i in ids)
    real_mm = float(np.hypot(bx - ax, by - ay))

    pa, pb = (detected_markers[i].mean(axis=0) for i in ids)
    pixel_px = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))

    return focal_length_px(pixel_px, real_mm, lens_to_markers_mm)


def lens_depth_mm(markers_depth_mm: float, lens_to_markers_mm: float, depth_sign: int) -> float:
    """Координата объектива вдоль оси глубины пола.

    depth_sign = +1, если координата вдоль оси глубины растёт по мере удаления от камеры.
    """
    return markers_depth_mm - depth_sign * lens_to_markers_mm


def contour_to_floor_mm(contour: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Контур верхней камеры (N,1,2) в пикселях -> (N,2) в мм плоскости пола."""
    points = contour.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def object_distances_mm(
    floor_points_mm: np.ndarray,
    lens_depth: float,
    depth_axis: DepthAxis = "y",
) -> tuple[float, float]:
    """Расстояния от объектива до ближней и дальней граней объекта. Возвращает (near, far)."""
    column = 0 if depth_axis == "x" else 1
    values = floor_points_mm[:, column]
    a = abs(float(values.min()) - lens_depth)
    b = abs(float(values.max()) - lens_depth)
    return min(a, b), max(a, b)


def height_mm_from_near_face(
    top_edge_px: float,
    bottom_edge_px: float,
    distance_near_mm: float,
    focal_px: float,
) -> float:
    """Высота объекта по его передней грани — той, что обращена к камере.

    Низ и верх передней грани находятся на ОДНОМ расстоянии от объектива, поэтому высота
    получается из одной разницы пикселей. Высота объектива над полом, положение оптического
    центра и небольшой наклон камеры при этом сокращаются — их не нужно ни мерить, ни хранить.

    Требование к монтажу поста: объектив боковой камеры должен стоять НИЖЕ самого низкого
    измеряемого объекта. Тогда камера не видит верхнюю грань, и верх силуэта — это верхнее
    ребро передней грани. Если камера окажется выше объекта, верхом силуэта станет дальнее
    ребро, и высота будет завышена (см. height_looks_inflated).
    """
    return abs(bottom_edge_px - top_edge_px) * distance_near_mm / focal_px


def height_looks_inflated(height_mm: float, lens_height_mm: float) -> bool:
    """True, если объект ниже объектива — камера видит его верхнюю грань и высота завышена."""
    return height_mm < lens_height_mm


def width_mm_from_pinhole(width_px: float, distance_near_mm: float, focal_px: float) -> float:
    """Ширина видимой грани: она обращена к камере, поэтому масштабируется ближним расстоянием."""
    return width_px * distance_near_mm / focal_px


def camera_nadir_mm(homography: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Точка пола прямо под верхней камерой — центр кадра, переведённый в мм."""
    width, height = image_size
    point = np.array([[[width / 2.0, height / 2.0]]], dtype=np.float32)
    return cv2.perspectiveTransform(point, homography).reshape(2)


def footprint_mm_without_side_faces(
    floor_points_mm: np.ndarray,
    nadir_mm: np.ndarray,
    height_mm: float,
    mount_height_mm: float,
) -> tuple[float, float]:
    """Габариты основания объекта по силуэту верхней камеры. Возвращает (длинная, короткая).

    Верхняя грань объекта на height_mm ближе к камере, поэтому в плоскости пола она растянута
    в m = mount / (mount - height) раз относительно точки надира. Силуэт — объединение
    основания и растянутой верхней грани: со стороны, обращённой к надиру, видно основание
    (координата уже верна), с дальней стороны — растянутую верхнюю грань (координату надо
    поделить на m). Если объект стоит по обе стороны от надира, боковых граней не видно вовсе
    и делить нужно обе стороны.

    Единый масштаб на весь силуэт (прежний correct_perspective) сжимает и ближнюю сторону
    тоже, и потому завышает габарит тем сильнее, чем дальше объект от центра кадра.
    """
    points = floor_points_mm.astype(np.float32).reshape(-1, 2)
    angle = np.deg2rad(cv2.minAreaRect(points)[2])
    axes = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    relative = points - np.asarray(nadir_mm, dtype=np.float32)
    magnification = (
        mount_height_mm / (mount_height_mm - height_mm)
        if 0.0 < height_mm < mount_height_mm
        else 1.0
    )

    sides = []
    for axis in axes:
        projected = relative @ axis
        low, high = float(projected.min()), float(projected.max())
        near = low / magnification if low < 0 else low
        far = high / magnification if high > 0 else high
        sides.append(far - near)
    return max(sides), min(sides)
