"""Калибровка плоскости основания груза для боковой камеры.

    python scripts/calibrate_floor_row.py --post 1 --height 100

Пост должен быть ОСТАНОВЛЕН: камеры нельзя открыть дважды.

Зачем. В силуэт боковой камеры связно с грузом попадает то, что лежит вокруг него на полу:
собственная тень, отражение в глянцевом полу, обрывок упаковки у ножки. Отбором контуров это не
лечится — с грузом они образуют одну фигуру. Лечится геометрией: ниже плоскости основания груза
быть не может. На стенде рулон высотой 100 мм читался как 125, потому что маска уходила на
156 px ниже основания.

Строка основания едет с расстоянием: `v = a + b/d`. Подставить сюда паспортные числа нельзя —
b = f*h требует высоты ОПТИЧЕСКОГО центра, а не корпуса, а a — середины кадра, которой нет при
малейшем наклоне камеры. На стенде подстановка дала 728 при измеренной 868. Поэтому оба числа
снимаются замером.

Как снимать. Взять предмет известной высоты (коробку, рулон — что угодно с плоским верхом),
измерить рулеткой и передать в --height. Дальше медленно двигать его по зоне ОТ камеры и К ней;
скрипт сам берёт замеры на разных расстояниях. Чем шире разброс расстояний, тем точнее b.

Метод. Для предмета высоты H на расстоянии d верх силуэта даёт `v_верх`, а высота в пикселях
равна `f*H/d`. Значит основание лежит на `v_верх + f*H/d`, и множитель f/d берётся из самого
замера: `(v_низ - v_верх) / H_изм`. Ошибка низа силуэта в результат при этом не входит — она
сокращается ровно настолько, насколько испортила H_изм.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.aruco_homography import compute_homography, detect_markers  # noqa: E402
from src.store import PostStore  # noqa: E402
from src.vision.perspective import (  # noqa: E402
    contour_to_floor_mm,
    focal_length_px_from_markers,
    lens_depth_mm,
    object_distances_mm,
)
from src.vision.segmentation import segment_pallet  # noqa: E402

MIN_DISTANCE_SPREAD_MM = 15.0  # насколько замер должен отличаться по расстоянию от уже взятых
MIN_SAMPLES = 4
GOOD_SPREAD_MM = 40.0  # меньший разброс расстояний даёт неустойчивое b


def open_camera(binding):
    capture = cv2.VideoCapture(binding.device_id, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, binding.resolution[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, binding.resolution[1])
    if not capture.isOpened():
        raise SystemExit(f"Камера {binding.device_id} не открылась. Пост остановлен?")
    return capture


def one_sample(post, cap_top, cap_side, true_height_mm):
    """(расстояние, строка основания) по одному кадру, либо None с причиной."""
    ok_top, frame_top = cap_top.read()
    ok_side, frame_side = cap_side.read()
    if not (ok_top and ok_side):
        return None, "кадр не пришёл"

    top, side = post.camera_top, post.camera_side
    homography = compute_homography(detect_markers(frame_top), top.aruco_marker_positions_mm)
    markers_side = detect_markers(frame_side)
    focal = focal_length_px_from_markers(markers_side, side.aruco_marker_positions_mm,
                                         side.reference_distance_mm)
    if homography is None or focal is None:
        return None, "не видны маркеры"

    contour_top = segment_pallet(frame_top, top.roi, top.background_hsv_lower,
                                 top.background_hsv_upper, top.min_contour_area_px,
                                 detect_markers(frame_top).values())
    contour_side = segment_pallet(frame_side, side.roi, side.background_hsv_lower,
                                  side.background_hsv_upper, side.min_contour_area_px,
                                  markers_side.values())
    if contour_top is None or contour_side is None:
        return None, "объект не найден"

    near_mm, _ = object_distances_mm(
        contour_to_floor_mm(contour_top, homography),
        lens_depth_mm(side.markers_depth_mm, side.reference_distance_mm, side.depth_sign),
        side.depth_axis,
    )
    _, top_px, _, height_px = cv2.boundingRect(contour_side)
    measured_mm = height_px * near_mm / focal
    if measured_mm <= 0:
        return None, "нулевая высота"

    # f/d берётся из самого замера, а не из focal/near: так ошибка низа силуэта сокращается.
    base_row = top_px + true_height_mm * height_px / measured_mm
    return (near_mm, base_row, measured_mm), None


def fit(samples):
    """Метод наименьших квадратов для v = a + b/d."""
    distances = np.array([s[0] for s in samples], dtype=np.float64)
    rows = np.array([s[1] for s in samples], dtype=np.float64)
    design = np.column_stack([np.ones_like(distances), 1.0 / distances])
    (a, b), *_ = np.linalg.lstsq(design, rows, rcond=None)
    residuals = rows - design @ np.array([a, b])
    return float(a), float(b), residuals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--post", default="1")
    parser.add_argument("--height", type=float, required=True, help="истинная высота предмета, мм")
    parser.add_argument("--url", default="http://127.0.0.1:8012")
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--apply", action="store_true", help="записать результат в настройки точки")
    args = parser.parse_args()

    post = PostStore("data/posts.json").get(args.post)
    if post is None:
        print(f"Точка {args.post} не найдена")
        return 1

    cap_top = open_camera(post.camera_top)
    cap_side = open_camera(post.camera_side)
    samples: list[tuple[float, float, float]] = []
    deadline = time.monotonic() + args.seconds
    last_note = 0.0

    print(f"Двигайте предмет высотой {args.height:.0f} мм по зоне — от камеры и к ней.")
    try:
        while time.monotonic() < deadline:
            sample, why = one_sample(post, cap_top, cap_side, args.height)
            now = time.monotonic()
            if sample is None:
                if now - last_note > 5:
                    print(f"  {why} ({deadline - now:.0f} с осталось)")
                    last_note = now
                continue
            distance, base_row, measured = sample
            if any(abs(distance - d) < MIN_DISTANCE_SPREAD_MM for d, _, _ in samples):
                continue
            samples.append(sample)
            print(f"  замер {len(samples)}: расстояние {distance:6.1f} мм, "
                  f"основание на строке {base_row:6.1f}, высота по силуэту {measured:6.1f} мм")
            last_note = now
    finally:
        cap_top.release()
        cap_side.release()

    if len(samples) < MIN_SAMPLES:
        print(f"\nСобрано {len(samples)} замеров, нужно минимум {MIN_SAMPLES}.")
        return 1

    spread = max(s[0] for s in samples) - min(s[0] for s in samples)
    a, b, residuals = fit(samples)
    print(f"\nfloor_row_a = {a:.1f}\nfloor_row_b = {b:.0f}")
    print(f"  разброс расстояний {spread:.0f} мм"
          f"{' — маловато, b неустойчиво' if spread < GOOD_SPREAD_MM else ''}")
    print(f"  остатки по строке: {np.abs(residuals).max():.1f} px худший, "
          f"{np.abs(residuals).mean():.1f} px средний")
    print(f"  строка основания: {a + b / 200:.0f} при 200 мм, {a + b / 400:.0f} при 400 мм")

    if not args.apply:
        print("\nЗапустите с --apply, чтобы записать это в настройки точки.")
        return 0

    side = json.loads(post.camera_side.model_dump_json())
    side["floor_row_a"], side["floor_row_b"] = round(a, 1), round(b, 0)
    request = urllib.request.Request(
        f"{args.url}/api/posts/{args.post}", data=json.dumps({"camera_side": side}).encode(),
        headers={"Content-Type": "application/json"}, method="PATCH")
    with urllib.request.urlopen(request) as response:
        written = json.load(response)["camera_side"]
    print(f"\nЗаписано: floor_row_a={written['floor_row_a']} floor_row_b={written['floor_row_b']}")
    print("Запустите точку заново — воркер поднимется с новым конфигом.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
