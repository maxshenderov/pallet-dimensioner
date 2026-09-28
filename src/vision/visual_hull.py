"""Вокселная резка по силуэтам (visual hull): габариты груза по нескольким камерам сверху.

Одна камера сверху принципиально не может измерить многоуровневый груз. Её силуэт говорит,
в каком НАПРАВЛЕНИИ от объектива находится объект, но не КАК ДАЛЕКО он по этому лучу. Ролик
⌀600×2000, стоящий на паллете 1200×800, с высоты 4 м даёт ровно тот же силуэт, что плоский
диск ⌀1292, лежащий на полу: верхняя грань ролика увеличена в 4000/(4000−2144) = 2.16 раза.
Габарит читается как 1292×1292 вместо 1200×800. Поправка на одну высоту (хоть с дальномера,
хоть из footprint_mm_without_side_faces) тут не спасает: у груза НЕСКОЛЬКО уровней, и каждому
нужна своя поправка.

Резка снимает неоднозначность геометрически. Измерительный объём режется на воксели; воксель
остаётся жив, только если попал внутрь силуэта ВО ВСЕХ камерах. Фантом на месте «раздутой»
верхней грани выживает у надирной камеры, но камера, стоящая сбоку по азимуту, смотрит на это
место сквозь пустоту и убивает его. Отсюда требование к посту: одна надирная камера + две
косые, разнесённые по азимуту на ~90°, — фантом, спрятанный в тени груза от одной косой
камеры, всегда открыт для другой.

Резка КОНСЕРВАТИВНА: воксель выживает при любом сомнении (шум фона, воксель вне кадра). Значит
все ошибки метода дают ЗАВЫШЕНИЕ габарита, а не занижение, — безопасное направление для WMS.

Остаточная ошибка метода — «юбка» под собственной проекцией груза: воксель на уровне пола
чуть за краем паллеты не режется ничем, потому что луч к нему сверху задевает саму паллету
выше. Её ширина = высота_паллеты × смещение_от_надира / высота_подвеса ≈ 22 мм на краю зоны
при паллете 144 мм и подвесе 4 м.

См. [[измерение_по_расстоянию]] — тот же по природе разбор для схемы с боковой камерой.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import cv2
import numpy as np


class HullBox(NamedTuple):
    """Габариты в порядке, принятом во всём сервисе: высота → ширина → глубина."""

    height_mm: float
    width_mm: float
    depth_mm: float


def voxel_grid(
    zone_mm: tuple[float, float],
    volume_height_mm: float,
    voxel_mm: float,
) -> np.ndarray:
    """Центры вокселей измерительного объёма. Возвращает (N, 3) в мм.

    Объём привязан к плоскости пола: X ∈ [−W/2, W/2], Y ∈ [−D/2, D/2], Z ∈ [0, H], начало
    координат — центр размеченной зоны (тот же, что задают углы красного прямоугольника).
    Точки — центры ячеек, поэтому крайние воксели отстоят от границы объёма на voxel_mm / 2.
    """
    if voxel_mm <= 0:
        raise ValueError("voxel_mm должен быть больше нуля")

    width_mm, depth_mm = zone_mm
    axes = []
    for extent, start in ((width_mm, -width_mm / 2), (depth_mm, -depth_mm / 2), (volume_height_mm, 0.0)):
        count = max(1, int(round(extent / voxel_mm)))
        axes.append(start + voxel_mm * (np.arange(count) + 0.5))

    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1).astype(np.float64)


def project_grid(
    grid: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Проекция вокселей в пиксели камеры. Возвращает (N, 2).

    Считается НЕ на каждом кадре: поза камеры меняется только когда сдвинулся красный
    прямоугольник. Между пересчётами резка — это просто индексация масок.
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)
    projected, _ = cv2.projectPoints(
        grid.reshape(-1, 1, 3), rvec, tvec, camera_matrix, np.asarray(dist_coeffs, dtype=np.float64)
    )
    return projected.reshape(-1, 2)


def carve(projections: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> np.ndarray:
    """Занятость вокселей: True, если воксель попал в маску объекта во всех камерах.

    Воксель, спроецировавшийся ЗА пределы кадра, считается ПУСТЫМ. Соблазнительно считать
    наоборот («не видели — значит не знаем, оставим на всякий случай»), но это даёт мусор:
    непросматриваемые углы объёма выживают всегда, и габарит вырождается в размер самого
    измерительного объёма — на проверочном ролике так получалась высота ровно 2200 мм при
    объёме высотой 2200 мм.

    Правильное место для этой проблемы — монтаж, а не расчёт: измерительный объём обязан
    целиком попадать в кадр КАЖДОЙ камеры. Это проверяется frame_coverage() один раз при
    настройке поста; при 100% выбор политики уже ни на что не влияет.
    """
    if len(projections) != len(masks):
        raise ValueError("Число проекций и масок должно совпадать")
    if not projections:
        raise ValueError("Нужна хотя бы одна камера")

    occupancy = np.ones(len(projections[0]), dtype=bool)
    for projection, mask in zip(projections, masks):
        height, width = mask.shape[:2]
        x = np.rint(projection[:, 0]).astype(np.int64)
        y = np.rint(projection[:, 1]).astype(np.int64)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)

        seen_as_object = np.zeros(len(projection), dtype=bool)
        seen_as_object[inside] = mask[y[inside], x[inside]] > 0
        occupancy &= seen_as_object
    return occupancy


def frame_coverage(projection: np.ndarray, image_size: tuple[int, int]) -> float:
    """Доля вокселей объёма, попадающих в кадр камеры, — диагностика расстановки.

    Меньше 1.0 означает, что часть измерительного объёма камере не видна и резка там не
    работает. Проверяется при монтаже поста, а не на каждом кадре. Требование к посту:
    100% у каждой камеры — иначе габарит занижается на невидимой части объёма.
    """
    width, height = image_size
    x, y = projection[:, 0], projection[:, 1]
    return float(((x >= 0) & (x < width) & (y >= 0) & (y < height)).mean())


def bounding_box_mm(grid: np.ndarray, occupancy: np.ndarray, voxel_mm: float) -> HullBox | None:
    """Габариты уцелевшего объёма. None, если объекта в зоне нет (штатная ситуация).

    Ширина и глубина — стороны минимального ПОВЁРНУТОГО прямоугольника: паллета редко стоит
    строго по осям разметки, а WMS нужен габарит самой паллеты, а не её проекции на разметку.
    Точки сетки — центры вокселей, поэтому к каждому измерению добавляется voxel_mm.
    """
    points = grid[occupancy]
    if len(points) < 3:
        return None

    (_, (a, b), _) = cv2.minAreaRect(points[:, :2].astype(np.float32))
    return HullBox(
        height_mm=float(points[:, 2].max()) + voxel_mm / 2,
        width_mm=max(a, b) + voxel_mm,
        depth_mm=min(a, b) + voxel_mm,
    )


def floor_area_seen_behind_volume(
    camera_position_mm: np.ndarray,
    zone_mm: tuple[float, float],
    volume_height_mm: float,
) -> tuple[float, float, float, float]:
    """Участок пола, который камера видит ПОЗАДИ груза. Возвращает (x_min, x_max, y_min, y_max).

    Косая камера смотрит на груз сверху-сбоку, поэтому фоном для верхней части груза служит
    не размеченная зона, а пол далеко за ней. Красить зелёным нужно именно этот участок —
    он заметно больше красного прямоугольника. Считается по восьми углам измерительного
    объёма: луч из камеры через угол продлевается до плоскости пола.
    """
    width_mm, depth_mm = zone_mm
    corners = np.array([
        [x, y, z]
        for x in (-width_mm / 2, width_mm / 2)
        for y in (-depth_mm / 2, depth_mm / 2)
        for z in (0.0, volume_height_mm)
    ], dtype=np.float64)

    camera = np.asarray(camera_position_mm, dtype=np.float64)
    # Точка на полу, куда приходит луч, продолженный за угол объёма: P + (C − P) · Pz / (Pz − Cz).
    denominator = corners[:, 2] - camera[2]
    scale = np.divide(corners[:, 2], denominator, out=np.zeros(len(corners)), where=denominator != 0)
    hits = corners + (camera - corners) * scale[:, None]

    return (
        float(min(hits[:, 0].min(), -width_mm / 2)),
        float(max(hits[:, 0].max(), width_mm / 2)),
        float(min(hits[:, 1].min(), -depth_mm / 2)),
        float(max(hits[:, 1].max(), depth_mm / 2)),
    )
