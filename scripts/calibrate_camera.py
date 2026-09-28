"""Калибровка объектива камеры поста по шахматной доске.

    python scripts/calibrate_camera.py --post 1 --camera top

Пост должен быть ОСТАНОВЛЕН: камеру нельзя открыть дважды.

Зачем это нужно. Без калибровки гомография верхней камеры считает одинаковые метки разными:
на посту 1 четыре одинаковых метки отобразились в 63.6 / 67.3 / 70.5 / 75.1 мм, и размер падал
ровно по мере удаления метки от центра кадра — бочкообразная дисторсия. Груз в середине зоны
меряется при этом точно (след 45x77 сошёлся в 0.4 мм), а тот же груз в углу зоны уедет на
проценты. Гомография дисторсию убрать не может: она точна для плоскости БЕЗ искажений.

Как снимать. Доску держат перед камерой и медленно водят: по всем четырём углам кадра, ближе и
дальше, с наклоном в обе стороны. Скрипт сам берёт кадр, когда доска нашлась и достаточно
отличается от уже принятых, — нажимать ничего не нужно. Углы кадра важнее центра: именно там
дисторсия и живёт, и по одному только центру она не определяется.

Доска: 9x6 ВНУТРЕННИХ углов (то есть 10x7 клеток), клетка 25 мм — обычная печать на A4,
наклеенная на что-то жёсткое. Волнистая доска испортит калибровку тише, чем её отсутствие.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.camera_calibrate import (  # noqa: E402
    MIN_VALID_IMAGES,
    calibrate_lens,
    save_intrinsics,
)
from src.store import PostStore  # noqa: E402

DETECT_SCALE = 0.5  # поиск доски на уменьшенной копии: на 1920x1080 полный поиск ~1 с на кадр
MIN_CENTRE_SHIFT_PX = 120  # насколько доска должна переехать, чтобы кадр считался новым
MIN_AREA_RATIO = 1.18  # либо переехать, либо заметно изменить размер (то есть наклон/дистанцию)


def is_new_view(centre, area, accepted) -> bool:
    """Кадр полезен, если доска стоит в новом месте ИЛИ под заметно другим углом.

    Пятнадцать снимков доски из одного положения дают ровно столько же сведений, сколько один:
    calibrateCamera найдёт коэффициенты, отлично описывающие эту точку кадра, и никакие другие.
    """
    for other_centre, other_area in accepted:
        moved = float(np.hypot(*(np.asarray(centre) - np.asarray(other_centre))))
        resized = max(area, other_area) / max(min(area, other_area), 1.0)
        if moved < MIN_CENTRE_SHIFT_PX and resized < MIN_AREA_RATIO:
            return False
    return True


def collect(device_id: int, resolution, board, shots: int, seconds: float, shots_dir: Path):
    capture = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    if not capture.isOpened():
        raise SystemExit(f"Камера {device_id} не открылась. Пост остановлен? "
                         f"POST /api/posts/{{id}}/stop")

    accepted: list[tuple] = []
    paths: list[str] = []
    deadline = time.monotonic() + seconds
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    last_report = 0.0

    print(f"Веду поиск доски {board[0]}x{board[1]} внутренних углов. "
          f"Нужно {shots} кадров, есть {seconds:.0f} с.")
    while len(paths) < shots and time.monotonic() < deadline:
        ok, frame = capture.read()
        if not ok:
            continue
        small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
        found, corners = cv2.findChessboardCorners(
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), board, flags)

        now = time.monotonic()
        if not found:
            if now - last_report > 5:
                print(f"  доска не видна ({deadline - now:.0f} с осталось)")
                last_report = now
            continue

        points = corners.reshape(-1, 2)
        centre = points.mean(axis=0) / DETECT_SCALE
        area = float(cv2.contourArea(cv2.convexHull(points))) / DETECT_SCALE ** 2
        if not is_new_view(centre, area, accepted):
            continue

        path = shots_dir / f"shot_{len(paths):02d}.png"
        cv2.imwrite(str(path), frame)
        paths.append(str(path))
        accepted.append((centre, area))
        print(f"  принят кадр {len(paths)}/{shots}: доска в ({centre[0]:.0f}, {centre[1]:.0f}), "
              f"площадь {area / 1000:.0f} тыс. px")
        last_report = now

    capture.release()
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--post", default="1")
    parser.add_argument("--camera", choices=("top", "side"), default="top")
    parser.add_argument("--shots", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--cols", type=int, default=9, help="внутренних углов по горизонтали")
    parser.add_argument("--rows", type=int, default=6, help="внутренних углов по вертикали")
    parser.add_argument("--square", type=float, default=25.0, help="сторона клетки, мм")
    parser.add_argument("--keep-shots", action="store_true", help="не удалять снимки после расчёта")
    args = parser.parse_args()

    post = PostStore("data/posts.json").get(args.post)
    if post is None:
        print(f"Точка {args.post} не найдена в data/posts.json")
        return 1
    binding = post.camera_top if args.camera == "top" else post.camera_side

    post_dir = Path("data/posts") / args.post
    post_dir.mkdir(parents=True, exist_ok=True)
    shots_dir = post_dir / f"calibration_shots_{args.camera}"
    shots_dir.mkdir(exist_ok=True)

    paths = collect(binding.device_id, binding.resolution, (args.cols, args.rows),
                    args.shots, args.seconds, shots_dir)
    if len(paths) < MIN_VALID_IMAGES:
        print(f"\nСобрано {len(paths)} кадров, нужно минимум {MIN_VALID_IMAGES}. "
              f"Калибровка не выполнена — снимки остались в {shots_dir}")
        return 1

    intrinsics = calibrate_lens(paths, (args.cols, args.rows), args.square, camera_id=args.camera)
    written = save_intrinsics(intrinsics, post_dir)

    k1, k2 = intrinsics.dist_coeffs[0], intrinsics.dist_coeffs[1]
    print(f"\nГотово: {written}")
    print(f"  ошибка перепроецирования {intrinsics.reprojection_error:.3f} px "
          f"({'хорошо' if intrinsics.reprojection_error < 1.0 else 'великовато — доска гнётся?'})")
    print(f"  фокусное {intrinsics.camera_matrix[0][0]:.0f} x {intrinsics.camera_matrix[1][1]:.0f} px, "
          f"оптический центр ({intrinsics.camera_matrix[0][2]:.0f}, {intrinsics.camera_matrix[1][2]:.0f})")
    print(f"  дисторсия k1={k1:+.4f} k2={k2:+.4f} "
          f"({'бочка' if k1 < 0 else 'подушка'})")
    print("\nВоркер подхватит файл при следующем запуске точки — перезапустите пост.")

    if not args.keep_shots:
        shutil.rmtree(shots_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
