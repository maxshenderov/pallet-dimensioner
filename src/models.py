"""Pydantic-модели событий измерения и конфигурации поста pallet-dimensioner."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class MeasurementEvent(BaseModel):
    """Событие измерения габаритов паллеты, отправляемое в POST /api/v1/pallet-measurements."""

    post_id: str
    timestamp: datetime
    length_mm: float
    width_mm: float
    height_mm: float
    cross_check_passed: bool
    cross_check_delta_mm: float
    samples_count: int = Field(gt=0)
    measurement_method: Literal["dual_webcam_ortho"] = "dual_webcam_ortho"


class CameraIntrinsics(BaseModel):
    """Результат калибровки объектива камеры (cv2.calibrateCamera)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_id: str
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    image_size: tuple[int, int]
    reprojection_error: float

    def as_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array(self.camera_matrix, dtype=np.float64),
            np.array(self.dist_coeffs, dtype=np.float64),
        )


class CameraConfig(BaseModel):
    """Общие поля конфигурации камеры (top/side)."""

    device_id: int
    resolution: tuple[int, int]
    roi: tuple[int, int, int, int]  # x, y, w, h
    aruco_marker_ids: list[int]
    aruco_marker_positions_mm: dict[int, tuple[float, float]]
    background_hsv_lower: tuple[int, int, int]
    background_hsv_upper: tuple[int, int, int]
    min_contour_area_px: int


class CameraTopConfig(CameraConfig):
    mount_height_mm: float


class CameraSideConfig(CameraConfig):
    reference_distance_mm: float
    floor_line_px: int


class StabilizerConfig(BaseModel):
    window_size: int = Field(gt=0)
    stddev_threshold_mm: float = Field(gt=0)


class CrossCheckConfig(BaseModel):
    tolerance_mm: float = Field(gt=0)


class NetworkConfig(BaseModel):
    retry_attempts: int = Field(gt=0)
    retry_backoff_seconds: float = Field(gt=0)
    pending_events_dir: str


class PostConfig(BaseModel):
    """Полная конфигурация поста измерения (config/post_config.yaml)."""

    post_id: str
    api_endpoint: str
    camera_top: CameraTopConfig
    camera_side: CameraSideConfig
    stabilizer: StabilizerConfig
    cross_check: CrossCheckConfig
    network: NetworkConfig


class MeasurementResult(BaseModel):
    """Стабилизированный (медианный) результат серии измерений."""

    length_mm: float
    width_mm: float
    height_mm: float
    samples_count: int
