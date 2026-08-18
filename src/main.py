"""Точка входа pallet-dimensioner: основной цикл измерения габаритов паллет."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .calibration.aruco_homography import compute_homography, detect_markers
from .calibration.camera_calibrate import load_intrinsics, undistort
from .capture.camera_stream import CameraStream, CameraUnavailableError, get_synchronized_frames
from .integration.api_client import send_measurement
from .models import MeasurementEvent, MeasurementResult, PostConfig
from .vision.dimensions import compute_footprint_mm, compute_height_and_side_mm, correct_perspective, cross_check
from .vision.segmentation import segment_pallet
from .vision.stabilizer import MeasurementStabilizer

logger = logging.getLogger(__name__)

CAMERA_OFFLINE_RETRY_SECONDS = 5.0
CALIBRATION_LOST_RETRY_SECONDS = 1.0


def load_config(path: str | Path) -> PostConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PostConfig(**data)


def build_event(config: PostConfig, result: MeasurementResult, passed: bool, delta_mm: float) -> MeasurementEvent:
    return MeasurementEvent(
        post_id=config.post_id,
        timestamp=datetime.now(timezone.utc),
        height_mm=result.height_mm,
        width_mm=result.length_mm,
        depth_mm=result.width_mm,
        cross_check_passed=passed,
        cross_check_delta_mm=delta_mm,
        samples_count=result.samples_count,
    )


def run(config_path: str | Path = "config/post_config.yaml") -> None:
    config = load_config(config_path)
    config_dir = Path(config_path).parent

    intrinsics_top = load_intrinsics(config_dir, "top")
    intrinsics_side = load_intrinsics(config_dir, "side")

    stream_top = CameraStream(config.camera_top.device_id, config.camera_top.resolution)
    stream_side = CameraStream(config.camera_side.device_id, config.camera_side.resolution)
    stabilizer = MeasurementStabilizer(config.stabilizer.window_size, config.stabilizer.stddev_threshold_mm)

    logger.info("Пост %s запущен", config.post_id)

    while True:
        try:
            frame_top, frame_side = get_synchronized_frames(stream_top, stream_side)
        except CameraUnavailableError as e:
            logger.error("camera_offline: %s", e)
            time.sleep(CAMERA_OFFLINE_RETRY_SECONDS)
            continue

        frame_top = undistort(frame_top, intrinsics_top)
        frame_side = undistort(frame_side, intrinsics_side)

        homography_top = compute_homography(detect_markers(frame_top), config.camera_top.aruco_marker_positions_mm)
        homography_side = compute_homography(detect_markers(frame_side), config.camera_side.aruco_marker_positions_mm)

        if homography_top is None or homography_side is None:
            logger.warning("calibration_lost: не все ArUco-маркеры найдены в кадре")
            stabilizer.reset()
            time.sleep(CALIBRATION_LOST_RETRY_SECONDS)
            continue

        contour_top = segment_pallet(
            frame_top,
            config.camera_top.roi,
            config.camera_top.background_hsv_lower,
            config.camera_top.background_hsv_upper,
            config.camera_top.min_contour_area_px,
        )
        contour_side = segment_pallet(
            frame_side,
            config.camera_side.roi,
            config.camera_side.background_hsv_lower,
            config.camera_side.background_hsv_upper,
            config.camera_side.min_contour_area_px,
        )

        if contour_top is None or contour_side is None:
            stabilizer.reset()
            continue

        footprint = compute_footprint_mm(contour_top, homography_top)
        height, side = compute_height_and_side_mm(contour_side, homography_side, config.camera_side.floor_line_px)
        # Гомография камеры 1 рассчитана относительно плоскости пола — референсное расстояние
        # для коррекции перспективы совпадает с высотой монтажа камеры над полом.
        length, width = correct_perspective(
            footprint, height, config.camera_top.mount_height_mm, config.camera_top.mount_height_mm
        )
        passed = cross_check((length, width), side, config.cross_check.tolerance_mm)
        delta_mm = min(abs(length - side), abs(width - side))

        stabilizer.add_sample(length, width, height)

        if stabilizer.is_stable():
            result = stabilizer.get_stable_result()
            event = build_event(config, result, passed, delta_mm)
            send_measurement(
                event,
                config.api_endpoint,
                config.network.retry_attempts,
                config.network.retry_backoff_seconds,
                config.network.pending_events_dir,
            )
            logger.info(
                "Измерение отправлено: Ш=%.0f Г=%.0f В=%.0f cross_check=%s",
                result.length_mm, result.width_mm, result.height_mm, passed,
            )
            stabilizer.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run()
