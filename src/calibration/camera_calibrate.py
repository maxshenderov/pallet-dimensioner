"""Разовая калибровка искажений объектива камеры по кадрам шахматной доски."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..models import CameraIntrinsics

CHESSBOARD_SIZE = (9, 6)  # внутренних углов (ширина, высота)
SQUARE_SIZE_MM = 25.0
MIN_VALID_IMAGES = 5


def calibrate_lens(
    image_paths: list[str],
    chessboard_size: tuple[int, int] = CHESSBOARD_SIZE,
    square_size_mm: float = SQUARE_SIZE_MM,
    camera_id: str = "cam",
) -> CameraIntrinsics:
    """Калибрует объектив по набору фото шахматной доски под разными углами."""
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    object_points = []
    image_points = []
    image_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Не удалось прочитать изображение: {path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, chessboard_size)
        if not found:
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(objp)
        image_points.append(corners)

    if len(object_points) < MIN_VALID_IMAGES:
        raise ValueError(
            f"Шахматная доска найдена только на {len(object_points)} из {len(image_paths)} фото, "
            f"нужно минимум {MIN_VALID_IMAGES}"
        )

    reprojection_error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    return CameraIntrinsics(
        camera_id=camera_id,
        camera_matrix=camera_matrix.tolist(),
        dist_coeffs=dist_coeffs.flatten().tolist(),
        image_size=image_size,
        reprojection_error=float(reprojection_error),
    )


def undistort(frame: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Применяет коррекцию дисторсии объектива к кадру."""
    camera_matrix, dist_coeffs = intrinsics.as_numpy()
    return cv2.undistort(frame, camera_matrix, dist_coeffs)


def save_intrinsics(intrinsics: CameraIntrinsics, config_dir: str | Path) -> Path:
    """Сохраняет результат калибровки в config/camera_intrinsics_{cam_id}.json."""
    path = Path(config_dir) / f"camera_intrinsics_{intrinsics.camera_id}.json"
    path.write_text(intrinsics.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_intrinsics(config_dir: str | Path, camera_id: str) -> CameraIntrinsics:
    """Загружает результат калибровки из config/camera_intrinsics_{cam_id}.json."""
    path = Path(config_dir) / f"camera_intrinsics_{camera_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return CameraIntrinsics(**data)
