"""Pydantic-модели событий измерения и конфигурации поста pallet-dimensioner."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
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


# ---------------------------------------------------------------------------
# Веб-слой: точки измерения (посты), создаваемые/управляемые через веб-сервис.
# В отличие от PostConfig (строгий YAML для headless-режима), Post рассчитан на
# быстрое создание через API с разумными дефолтами — тонкая настройка ROI/HSV/
# ArUco делается уже после создания через PATCH.
# ---------------------------------------------------------------------------

DEFAULT_TOP_MARKER_POSITIONS_MM: dict[int, tuple[float, float]] = {
    1: (0.0, 0.0), 2: (1200.0, 0.0), 3: (1200.0, 800.0), 4: (0.0, 800.0),
}
DEFAULT_SIDE_MARKER_POSITIONS_MM: dict[int, tuple[float, float]] = {
    5: (0.0, 0.0), 6: (1000.0, 0.0),
}


class PostStatus(str, Enum):
    stopped = "stopped"
    starting = "starting"
    running = "running"
    camera_offline = "camera_offline"
    calibration_lost = "calibration_lost"


class CameraBinding(BaseModel):
    """Привязка камеры к точке — с дефолтами, чтобы точку можно было создать одним вызовом."""

    device_id: int
    resolution: tuple[int, int] = (1920, 1080)
    roi: tuple[int, int, int, int] = (0, 0, 1920, 1080)
    aruco_marker_ids: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    aruco_marker_positions_mm: dict[int, tuple[float, float]] = Field(
        default_factory=lambda: dict(DEFAULT_TOP_MARKER_POSITIONS_MM)
    )
    background_hsv_lower: tuple[int, int, int] = (0, 0, 0)
    background_hsv_upper: tuple[int, int, int] = (180, 60, 90)
    min_contour_area_px: int = 5000


class CameraTopBinding(CameraBinding):
    mount_height_mm: float = 3200.0


class CameraSideBinding(CameraBinding):
    aruco_marker_ids: list[int] = Field(default_factory=lambda: [5, 6])
    aruco_marker_positions_mm: dict[int, tuple[float, float]] = Field(
        default_factory=lambda: dict(DEFAULT_SIDE_MARKER_POSITIONS_MM)
    )
    reference_distance_mm: float = 2500.0
    floor_line_px: int = 980


class Post(BaseModel):
    """Точка измерения: имя + привязанные камеры + весы (scale_api) + опционально WMS-эндпоинт."""

    id: str
    name: str
    camera_top: CameraTopBinding
    camera_side: CameraSideBinding
    scale_api_url: str | None = None
    wms_endpoint: str | None = None
    stabilizer_window_size: int = Field(default=12, gt=0)
    stabilizer_stddev_threshold_mm: float = Field(default=15.0, gt=0)
    cross_check_tolerance_mm: float = Field(default=30.0, gt=0)
    created_at: datetime


class PostCreateRequest(BaseModel):
    """Минимальные данные для создания точки — остальное берётся по дефолту."""

    name: str
    camera_top_device_id: int
    camera_side_device_id: int
    scale_api_url: str | None = None
    wms_endpoint: str | None = None


class PostUpdateRequest(BaseModel):
    """Частичное обновление точки — калибровочные поля правятся уже после создания."""

    name: str | None = None
    camera_top: CameraTopBinding | None = None
    camera_side: CameraSideBinding | None = None
    scale_api_url: str | None = None
    wms_endpoint: str | None = None
    stabilizer_window_size: int | None = Field(default=None, gt=0)
    stabilizer_stddev_threshold_mm: float | None = Field(default=None, gt=0)
    cross_check_tolerance_mm: float | None = Field(default=None, gt=0)


class WeightReading(BaseModel):
    """Показание весов (формат scale_api GET /api/weight)."""

    ok: bool
    value: float | None = None
    unit: str | None = None
    stable: bool | None = None


class PostState(BaseModel):
    """Живое состояние точки — то, что видит рабочее место оператора."""

    post_id: str
    status: PostStatus = PostStatus.stopped
    length_mm: float | None = None
    width_mm: float | None = None
    height_mm: float | None = None
    cross_check_passed: bool | None = None
    stable: bool = False
    weight: WeightReading | None = None
    updated_at: datetime | None = None
    error: str | None = None
