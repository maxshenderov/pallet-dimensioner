"""Расчёт поста с камерами сверху без камер и без цеха.

Отвечает цифрами на вопросы, которые иначе решаются покупкой железа и покраской пола:
куда вешать косые камеры, сколько нужно камер, какую площадь красить зелёным, какая
остаточная ошибка у метода и сколько времени занимает резка.

    python -m scripts.simulate_hull                 # проверочный случай: ролик на паллете
    python -m scripts.simulate_hull --bench         # то же в масштабе 1:8 (стенд 50 см)
    python -m scripts.simulate_hull --sweep         # перебор выноса косых камер
    python -m scripts.simulate_hull --cameras 1     # сколько врёт одна камера сверху
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src.vision.synthetic import (
    look_at_pose,
    pallet_with_roll,
    pinhole_matrix,
    render_silhouette,
    two_level_load,
)
from src.vision.visual_hull import (
    bounding_box_mm,
    carve,
    floor_area_seen_behind_volume,
    frame_coverage,
    project_grid,
    voxel_grid,
)

IMAGE_SIZE = (1920, 1080)
HORIZONTAL_FOV_DEG = 56.0


def camera_positions(mount_mm: float, oblique_offset_mm: float, count: int) -> list[tuple[str, np.ndarray]]:
    """Надир + косые камеры, разнесённые по азимуту на 90°."""
    cameras = [("A надир", np.array([0.0, 0.0, mount_mm]))]
    if count >= 2:
        cameras.append(("B по глубине", np.array([0.0, -oblique_offset_mm, mount_mm])))
    if count >= 3:
        cameras.append(("C по ширине", np.array([-oblique_offset_mm, 0.0, mount_mm])))
    return cameras[:count]


def measure(scene, zone_mm, volume_height_mm, voxel_mm, cameras, fov_deg=HORIZONTAL_FOV_DEG) -> tuple:
    """Полный прогон: рендер силуэтов -> проекция сетки -> резка -> габарит."""
    camera_matrix = pinhole_matrix(IMAGE_SIZE, fov_deg)
    target = np.array([0.0, 0.0, volume_height_mm / 2])

    grid = voxel_grid(zone_mm, volume_height_mm, voxel_mm)
    masks, projections, coverage = [], [], {}
    for name, position in cameras:
        rvec, tvec = look_at_pose(position, target)
        masks.append(render_silhouette(scene, IMAGE_SIZE, rvec, tvec, camera_matrix))
        projections.append(project_grid(grid, rvec, tvec, camera_matrix))
        coverage[name] = frame_coverage(projections[-1], IMAGE_SIZE)

    started = time.perf_counter()
    occupancy = carve(projections, masks)
    box = bounding_box_mm(grid, occupancy, voxel_mm)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return box, len(grid), elapsed_ms, coverage


def report(title, result, truth) -> None:
    box, voxels, elapsed_ms, coverage = result
    seen = "  ".join(f"{name.split()[0]} {value:.0%}" for name, value in coverage.items())
    print(f"\n{title}")
    print(f"  вокселей {voxels:,}   резка {elapsed_ms:.0f} мс   объём в кадре: {seen}")
    if box is None:
        print("  объект не найден")
        return
    for name, measured, true_value in zip(("высота", "ширина", "глубина"), box, truth):
        print(f"  {name:<8} {measured:8.0f}   истина {true_value:6.0f}   ошибка {measured - true_value:+7.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true", help="масштаб стенда 1:8, подвес 500 мм")
    parser.add_argument("--cameras", type=int, default=3, help="сколько камер участвует в резке")
    parser.add_argument("--sweep", action="store_true", help="перебрать вынос косых камер")
    parser.add_argument("--fov", action="store_true", help="перебрать поле зрения объектива")
    parser.add_argument("--voxel", type=float, default=None, help="размер вокселя, мм")
    args = parser.parse_args()

    scale = 1 / 8 if args.bench else 1.0
    mount_mm = 4000.0 * scale
    zone_mm = (2000.0 * scale, 1500.0 * scale)
    volume_height_mm = 2200.0 * scale
    voxel_mm = args.voxel if args.voxel else 20.0 * scale
    oblique_offset_mm = 2500.0 * scale

    print(f"Масштаб 1:{1 / scale:.0f}   подвес {mount_mm:.0f} мм   зона {zone_mm[0]:.0f}×{zone_mm[1]:.0f} мм"
          f"   воксель {voxel_mm:.1f} мм")

    truth = (2144.0 * scale, 1200.0 * scale, 800.0 * scale)

    if args.fov:
        print("\nПоле зрения объектива (вынос косых камер фиксирован):")
        cameras = camera_positions(mount_mm, oblique_offset_mm, 3)
        for fov in (56.0, 65.0, 70.0, 78.0, 90.0, 100.0, 110.0):
            box, _, _, coverage = measure(
                pallet_with_roll(scale), zone_mm, volume_height_mm, voxel_mm, cameras, fov
            )
            errors = " ".join(f"{m - t:+6.0f}" for m, t in zip(box, truth)) if box else " нет объекта  "
            seen = " ".join(f"{v:.0%}" for v in coverage.values())
            print(f"  {fov:5.0f}°   ошибка В/Ш/Г {errors}   объём в кадре {seen}")
        return

    if args.sweep:
        print("\nВынос косых камер от центра зоны:")
        cameras_fov = 110.0
        for offset in (1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0):
            cameras = camera_positions(mount_mm, offset * scale, 3)
            box, _, _, coverage = measure(
                pallet_with_roll(scale), zone_mm, volume_height_mm, voxel_mm, cameras, cameras_fov
            )
            errors = " ".join(f"{m - t:+6.0f}" for m, t in zip(box, truth)) if box else " нет объекта  "
            seen = " ".join(f"{v:.0%}" for v in coverage.values())
            print(f"  {offset * scale:6.0f} мм   ошибка В/Ш/Г {errors}   объём в кадре {seen}")
        return

    cameras = camera_positions(mount_mm, oblique_offset_mm, args.cameras)
    print("Камеры: " + ", ".join(f"{name} {tuple(int(v) for v in pos)}" for name, pos in cameras))

    report(
        f"Ролик ⌀{600 * scale:.0f}×{2000 * scale:.0f} на паллете ({args.cameras} кам.)",
        measure(pallet_with_roll(scale), zone_mm, volume_height_mm, voxel_mm, cameras),
        (2144.0 * scale, 1200.0 * scale, 800.0 * scale),
    )
    report(
        f"Двухуровневый груз ({args.cameras} кам.)",
        measure(two_level_load(scale), zone_mm, volume_height_mm, voxel_mm, cameras),
        (1800.0 * scale, 1200.0 * scale, 800.0 * scale),
    )

    print("\nЗелёный пол — участок, который камеры видят позади груза:")
    for name, position in cameras:
        x_min, x_max, y_min, y_max = floor_area_seen_behind_volume(position, zone_mm, volume_height_mm)
        print(f"  {name:<14} X {x_min:8.0f} .. {x_max:7.0f}   Y {y_min:8.0f} .. {y_max:7.0f} мм")


if __name__ == "__main__":
    main()
