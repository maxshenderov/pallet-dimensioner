#!/usr/bin/env python
"""Прогон серии измерений против эталонного объекта известных размеров, вердикт пригодности."""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.aruco_homography import compute_homography, detect_markers
from src.calibration.camera_calibrate import load_intrinsics, undistort
from src.capture.camera_stream import CameraStream, get_synchronized_frames
from src.main import load_config
from src.vision.dimensions import compute_footprint_mm, compute_height_and_side_mm, correct_perspective
from src.vision.segmentation import segment_pallet

BIAS_TOLERANCE_MM = 30.0
STDDEV_TOLERANCE_MM = 15.0


@dataclass
class ReferenceObject:
    length_mm: float
    width_mm: float
    height_mm: float


def run(config_path: str, reference: ReferenceObject, samples: int) -> None:
    config = load_config(config_path)
    config_dir = Path(config_path).parent
    intrinsics_top = load_intrinsics(config_dir, "top")
    intrinsics_side = load_intrinsics(config_dir, "side")

    stream_top = CameraStream(config.camera_top.device_id, config.camera_top.resolution)
    stream_side = CameraStream(config.camera_side.device_id, config.camera_side.resolution)

    lengths: list[float] = []
    widths: list[float] = []
    heights: list[float] = []

    print(
        f"Снимаю {samples} измерений эталона "
        f"{reference.length_mm:.0f}x{reference.width_mm:.0f}x{reference.height_mm:.0f} мм..."
    )

    while len(lengths) < samples:
        frame_top, frame_side = get_synchronized_frames(stream_top, stream_side)
        frame_top = undistort(frame_top, intrinsics_top)
        frame_side = undistort(frame_side, intrinsics_side)

        homography_top = compute_homography(detect_markers(frame_top), config.camera_top.aruco_marker_positions_mm)
        homography_side = compute_homography(detect_markers(frame_side), config.camera_side.aruco_marker_positions_mm)
        if homography_top is None or homography_side is None:
            continue

        contour_top = segment_pallet(
            frame_top, config.camera_top.roi,
            config.camera_top.background_hsv_lower, config.camera_top.background_hsv_upper,
            config.camera_top.min_contour_area_px,
        )
        contour_side = segment_pallet(
            frame_side, config.camera_side.roi,
            config.camera_side.background_hsv_lower, config.camera_side.background_hsv_upper,
            config.camera_side.min_contour_area_px,
        )
        if contour_top is None or contour_side is None:
            continue

        footprint = compute_footprint_mm(contour_top, homography_top)
        height, _side = compute_height_and_side_mm(contour_side, homography_side, config.camera_side.floor_line_px)
        length, width = correct_perspective(
            footprint, height, config.camera_top.mount_height_mm, config.camera_top.mount_height_mm
        )

        lengths.append(length)
        widths.append(width)
        heights.append(height)
        print(f"  [{len(lengths)}/{samples}] L={length:.1f} W={width:.1f} H={height:.1f}")

    overall_ok = True
    overall_ok &= _report("Длина", lengths, reference.length_mm)
    overall_ok &= _report("Ширина", widths, reference.width_mm)
    overall_ok &= _report("Высота", heights, reference.height_mm)
    print(f"\nИтоговый вердикт: {'ПРИГОДНО' if overall_ok else 'НЕ ПРИГОДНО'}")


def _report(name: str, values: list[float], expected_mm: float) -> bool:
    mean = statistics.mean(values)
    bias = mean - expected_mm
    stddev = statistics.pstdev(values)
    ok = abs(bias) <= BIAS_TOLERANCE_MM and stddev <= STDDEV_TOLERANCE_MM
    verdict = "ПРИГОДНО" if ok else "НЕ ПРИГОДНО"
    print(f"{name}: среднее={mean:.1f} мм, смещение={bias:+.1f} мм, std={stddev:.1f} мм -> {verdict}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка точности измерения против эталонного объекта")
    parser.add_argument("--config", default="config/post_config.yaml")
    parser.add_argument("--length", type=float, required=True, help="Эталонная длина, мм")
    parser.add_argument("--width", type=float, required=True, help="Эталонная ширина, мм")
    parser.add_argument("--height", type=float, required=True, help="Эталонная высота, мм")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    run(args.config, ReferenceObject(args.length, args.width, args.height), args.samples)
