"""Оценка дисторсии объектива по калибровочным меткам — без шахматной доски.

    python scripts/calibrate_distortion.py --post 1 --camera top
    python scripts/calibrate_distortion.py --post 1 --camera top --apply

Пост должен быть ОСТАНОВЛЕН.

Метки на посту напечатаны одного размера и лежат в плоскости пола. Значит после верной
гомографии каждая их сторона обязана дать одно и то же число миллиметров. На посту 1 они дали
75.1 / 70.5 / 67.3 / 63.6, и размер падал ровно по мере удаления метки от центра кадра —
бочкообразная дисторсия, 18 % разброса.

Здесь подбирается ОДИН коэффициент k1 так, чтобы этот разброс стал минимальным. Мера и лекарство
совпадают, поэтому результат сразу видно: печатается разброс до и после.

ЧЕГО ЭТОТ СПОСОБ НЕ ДАЁТ. Шахматная доска даёт полный набор: два фокусных, оптический центр, k1,
k2, k3, тангенциальные. Здесь оптический центр принят за середину кадра, тангенциальные — нулём,
подбирается только k1. Этого хватает, чтобы снять основную часть бочки, и не хватает, чтобы
ручаться за края кадра и за смещённый объектив. Когда доска будет — снимайте ей
(scripts/calibrate_camera.py), это остаётся правильным способом.

Метрика — разброс по ВСЕМ шестнадцати сторонам (4 метки x 4 стороны), а не по четырём средним:
метка на полу после верной гомографии обязана быть квадратом, и перекос сторон внутри метки —
такая же ошибка, как разница между метками.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.aruco_homography import compute_homography, detect_markers  # noqa: E402
from src.calibration.camera_calibrate import save_intrinsics  # noqa: E402
from src.models import CameraIntrinsics  # noqa: E402
from src.store import PostStore  # noqa: E402

FRAMES = 5  # усреднение по нескольким кадрам: детектор углов шумит на доли пикселя
K1_RANGE = (-0.60, 0.20)
K1_STEPS = 161
REFINE_PASSES = 3


def grab(binding, frames: int):
    capture = cv2.VideoCapture(binding.device_id, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, binding.resolution[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, binding.resolution[1])
    if not capture.isOpened():
        raise SystemExit(f"Камера {binding.device_id} не открылась. Пост остановлен?")
    try:
        detections = []
        shape = None
        for _ in range(frames * 2):
            ok, frame = capture.read()
            if not ok:
                continue
            shape = frame.shape
            found = detect_markers(frame)
            if set(binding.aruco_marker_ids).issubset(found):
                detections.append({i: found[i] for i in binding.aruco_marker_ids})
            if len(detections) >= frames:
                break
    finally:
        capture.release()

    if not detections:
        raise SystemExit("Метки в кадре не найдены — калибровать не по чему")
    averaged = {i: np.mean([d[i] for d in detections], axis=0) for i in detections[0]}
    return averaged, (shape[1], shape[0])


def focal_guess(markers, positions_mm, plane_distance_mm) -> float:
    """Фокусное по паре меток, разнесённых в плоскости пола на известное расстояние."""
    ids = sorted(positions_mm)
    best = max(((a, b) for i, a in enumerate(ids) for b in ids[i + 1:]),
               key=lambda pair: np.hypot(*(np.subtract(positions_mm[pair[1]], positions_mm[pair[0]]))))
    a, b = best
    real_mm = float(np.hypot(*(np.subtract(positions_mm[b], positions_mm[a]))))
    pixel = float(np.linalg.norm(markers[b].mean(axis=0) - markers[a].mean(axis=0)))
    return pixel * plane_distance_mm / real_mm


def side_lengths_mm(markers, positions_mm, camera_matrix, k1):
    """Стороны всех меток в мм после снятия дисторсии и построения гомографии."""
    distortion = np.array([k1, 0.0, 0.0, 0.0, 0.0])
    fixed = {}
    for marker_id, corners in markers.items():
        points = np.asarray(corners, np.float32).reshape(-1, 1, 2)
        fixed[marker_id] = cv2.undistortPoints(points, camera_matrix, distortion,
                                               P=camera_matrix).reshape(-1, 2)

    homography = compute_homography(fixed, positions_mm)
    if homography is None:
        return None

    sides = []
    for corners in fixed.values():
        floor = cv2.perspectiveTransform(
            corners.astype(np.float32).reshape(-1, 1, 2), homography).reshape(-1, 2)
        sides += [float(np.hypot(*(floor[(i + 1) % 4] - floor[i]))) for i in range(4)]
    return np.array(sides)


def spread(sides) -> float:
    return float(sides.std() / sides.mean()) if sides is not None else float("inf")


def search(markers, positions_mm, focal, centre, free_centre: bool):
    """Покоординатный спуск по k1 и, если разрешено, по оптическому центру.

    Центр приходится отдавать в подбор: радиальная картина центрирована на оптической оси, а она
    у дешёвой камеры смещена от середины кадра. С центром, прибитым к середине, k1 подгонять не
    к чему — на посту 1 разброс падал с 6.5 % всего до 5.6 %.
    """
    state = [0.0, float(centre[0]), float(centre[1])]
    spans = [(K1_RANGE[1] - K1_RANGE[0]) / 4, 200.0, 200.0]

    def cost(vector):
        matrix = np.array([[focal, 0.0, vector[1]], [0.0, focal, vector[2]], [0.0, 0.0, 1.0]])
        return spread(side_lengths_mm(markers, positions_mm, matrix, vector[0]))

    for _ in range(REFINE_PASSES * 3):
        for axis in range(3 if free_centre else 1):
            grid = np.linspace(state[axis] - spans[axis], state[axis] + spans[axis], K1_STEPS)
            costs = []
            for value in grid:
                trial = list(state)
                trial[axis] = float(value)
                costs.append(cost(trial))
            state[axis] = float(grid[int(np.argmin(costs))])
        spans = [s / 3 for s in spans]
    return state


def report(title, sides, markers, centre):
    print(f"  {title}: стороны {sides.min():.1f}..{sides.max():.1f} мм, "
          f"разброс {100 * spread(sides):.1f}%")
    for n, (marker_id, corners) in enumerate(sorted(markers.items())):
        points = np.asarray(corners).reshape(-1, 2)
        radius = float(np.linalg.norm(points.mean(axis=0) - centre))
        in_px = np.mean([np.hypot(*(points[(i + 1) % 4] - points[i])) for i in range(4)])
        block = sides[n * 4:(n + 1) * 4]
        print(f"    метка {marker_id}: {block.mean():5.1f} мм, {in_px:5.1f} px, "
              f"от центра кадра {radius:4.0f} px")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--post", default="1")
    parser.add_argument("--camera", choices=("top", "side"), default="top")
    parser.add_argument("--apply", action="store_true", help="сохранить camera_intrinsics_*.json")
    parser.add_argument("--fixed-centre", action="store_true",
                        help="не подбирать оптический центр, оставить серединой кадра")
    args = parser.parse_args()

    post = PostStore("data/posts.json").get(args.post)
    if post is None:
        print(f"Точка {args.post} не найдена")
        return 1
    binding = post.camera_top if args.camera == "top" else post.camera_side
    if len(binding.aruco_marker_ids) < 3:
        print(f"Камере {args.camera} видны только {len(binding.aruco_marker_ids)} метки — "
              f"гомографию по ним не построить. Способ работает только для верхней камеры.")
        return 1

    markers, image_size = grab(binding, FRAMES)
    centre = np.array([image_size[0] / 2.0, image_size[1] / 2.0])
    plane_mm = getattr(binding, "mount_height_mm", None) or binding.reference_distance_mm
    focal = focal_guess(markers, binding.aruco_marker_positions_mm, plane_mm)
    camera_matrix = np.array([[focal, 0.0, centre[0]], [0.0, focal, centre[1]], [0.0, 0.0, 1.0]])
    print(f"фокусное по меткам {focal:.0f} px, оптический центр принят серединой кадра")

    before = side_lengths_mm(markers, binding.aruco_marker_positions_mm, camera_matrix, 0.0)
    if before is None:
        print("Гомография не строится — проверьте таблицу меток")
        return 1
    print("\nбез поправки:")
    report("сейчас", before, markers, centre)

    k1, cx, cy = search(markers, binding.aruco_marker_positions_mm, focal, centre,
                        free_centre=not args.fixed_centre)
    camera_matrix = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    after = side_lengths_mm(markers, binding.aruco_marker_positions_mm, camera_matrix, k1)
    shape = "бочка" if k1 < 0 else "подушка"
    print(f"\nподобрано k1 = {k1:+.4f} ({shape}), оптический центр ({cx:.0f}, {cy:.0f}) "
          f"при середине кадра ({centre[0]:.0f}, {centre[1]:.0f}):")
    report("после", after, markers, centre)

    gain = spread(before) / max(spread(after), 1e-9)
    print(f"\nразброс упал в {gain:.1f} раза: {100 * spread(before):.1f}% -> {100 * spread(after):.1f}%")
    if spread(after) > 0.03:
        print("  Остаток больше 3% — одним коэффициентом эта оптика не описывается, нужна доска.")

    if not args.apply:
        print("\nЗапустите с --apply, чтобы сохранить. Воркер подхватит файл при следующем "
              "запуске точки.")
        return 0

    intrinsics = CameraIntrinsics(
        camera_id=args.camera,
        camera_matrix=camera_matrix.tolist(),
        dist_coeffs=[k1, 0.0, 0.0, 0.0, 0.0],
        image_size=image_size,
        reprojection_error=float(spread(after)),
    )
    written = save_intrinsics(intrinsics, Path("data/posts") / args.post)
    print(f"\nСохранено: {written}")
    print("ВНИМАНИЕ: оценка по меткам, не по доске: подобраны только k1 и оптический центр, "
          "k2/k3 и тангенциальные оставлены нулями.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
