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


def floor_row_px(a: float | None, b: float | None, distance_mm: float) -> float | None:
    """Строка кадра, где проходит плоскость основания груза. None, если пост не откалиброван.

    У камеры с горизонтальной осью точка на высоте Y над полом даёт строку
    `v = v0 - f*(Y - h)/d`, где h — высота объектива, d — расстояние вдоль оси. При Y = 0 это
    `v = v0 + f*h/d`, то есть ровно `a + b/d` с a = v0 и b = f*h. Оба числа снимаются один раз:
    объект известной высоты ставится в два места зоны (scripts/calibrate_floor_row.py).

    Считать их из конфига напрямую нельзя. Формально b = f*h, но h — высота ОПТИЧЕСКОГО ЦЕНТРА,
    а не корпуса, и a — не середина кадра, если камера хоть чуть наклонена. На стенде подстановка
    паспортных чисел дала строку 728 при измеренной 868.
    """
    if a is None or b is None or distance_mm <= 0:
        return None
    return a + b / distance_mm


def clip_below_floor(contour: np.ndarray, floor_row: float | None) -> np.ndarray:
    """Обрезает контур по плоскости основания: ниже неё груза быть не может.

    Это единственная защита от того, что в силуэт попадает НЕ груз, а то, что лежит на полу
    вокруг него, и попадает связно — тень под грузом, отражение в глянцевом полу, обрывок
    упаковки. Ни один отбор контуров тут не спасает: всё это одна фигура вместе с грузом.
    Замер на стенде: рулон высотой 100 мм читался как 125, потому что маска продолжалась на
    156 px ниже основания через собственную тень на лежащий рядом малярный скотч.

    Обрезка только СНИЗУ и только вниз: если маска не дотянулась до основания (тёмный низ груза
    не выделился), контур не достраивается. Так занижение остаётся видимым, а не маскируется.
    """
    if floor_row is None:
        return contour
    clipped = contour.copy()
    rows = clipped[..., 1]
    np.minimum(rows, np.float32(floor_row).astype(rows.dtype), out=rows)
    return clipped


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
    """Габариты основания объекта по силуэту верхней камеры. Возвращает (ширина, глубина).

    **Ширина — вдоль оси x пола, глубина — вдоль оси y.** Оси пола задаёт таблица
    `aruco_marker_positions_mm` верхней камеры: на посту 1 метки 2 и 1 стоят в (0,0) и (240,0),
    то есть линия «метка 2 -> метка 1» и есть ширина; метки 2 и 4 в (0,0) и (0,185) задают
    глубину. Кто заполняет таблицу меток — тот и решает, что на посту считается шириной.

    Боковая камера в определение не входит вовсе. Две прежние версии этого места ошибались
    именно здесь: сначала ширина была просто ДЛИННОЙ стороной следа (тогда поворот груза на 90°
    не менял ни одной цифры), потом — стороной, обращённой к боковой камере (на посту 1 это
    даёт ровно обратные названия). Ошибка выглядела как «сервис путает ширину с глубиной», и
    обучаемая поправка её маскировала: перестановку осей регрессия выучивает и молча ломается,
    как только груз повернут иначе, чем при обучении.

    Стороны считаются вдоль СОБСТВЕННЫХ осей следа (`minAreaRect`), а не вдоль осей пола: у
    повёрнутого груза проекция на оси пола дала бы описанный прямоугольник вместо его сторон.
    По осям пола выбирается только НАЗВАНИЕ: какая из двух осей следа ближе к x — та ширина.

    Верхняя грань объекта на height_mm ближе к камере, поэтому в плоскости пола она растянута
    в m = mount / (mount - height) раз относительно точки надира. Весь силуэт делится на m —
    то есть силуэт считается изображением ВЕРХНЕЙ ГРАНИ, а не объединения грани с основанием.

    Геометрически основание тоже попадает в силуэт: с той стороны, где надир снаружи следа,
    ближний край объединения принадлежит основанию, и делить его не нужно. Так и было написано
    сначала. На стенде это давало систематический промах в четверть по короткой стороне: у
    коробки 122 x 78 x 44, стоящей вертикально, силуэт по короткой оси 62.8 мм при увеличении
    1.409; деление всего силуэта даёт 44.5 мм при истинных 44, а «ближний край — основание» —
    32.6. Причина в том, что ближняя боковая грань обращена от света, выходит тёмной и в маску
    не попадает: до основания силуэт просто не доходит. По длинной оси надир был внутри следа,
    обе формулы совпадали — потому и промахивалась ровно одна сторона.

    Если освещение поменяется и боковые грани начнут попадать в маску, эта формула начнёт
    занижать. Признак — габарит поехал вниз тем сильнее, чем выше груз.
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
        sides.append((float(projected.max()) - float(projected.min())) / magnification)

    # Ось следа, которая ближе к оси x пола, и есть ширина. Оси следа взаимно перпендикулярны,
    # поэтому достаточно сравнить с x: та, у которой |cos| больше, лежит вдоль x.
    along_x = int(np.argmax([abs(float(axis[0])) for axis in axes]))
    return sides[along_x], sides[1 - along_x]
