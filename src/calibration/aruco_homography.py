"""Расчёт гомографии «пиксели кадра -> мм в плоскости сцены» по ArUco-маркерам."""
from __future__ import annotations

import cv2
import numpy as np

ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


def detect_markers(frame: np.ndarray, dictionary: str = "DICT_5X5_100") -> dict[int, np.ndarray]:
    """Находит ArUco-маркеры в кадре. Возвращает {id: 4x2 массив углов маркера в пикселях}."""
    dict_id = ARUCO_DICTIONARIES.get(dictionary)
    if dict_id is None:
        raise ValueError(f"Неизвестный словарь ArUco: {dictionary}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return {}

    return {int(np.asarray(marker_id).item()): corners[i][0] for i, marker_id in enumerate(ids)}


def compute_homography(
    detected_markers: dict[int, np.ndarray],
    known_marker_positions_mm: dict[int, tuple[float, float]],
) -> np.ndarray | None:
    """
    Считает гомографию по центрам маркеров конфигурации поста.

    Требует, чтобы ВЕСЬ набор маркеров из known_marker_positions_mm был найден в кадре —
    если хотя бы один отсутствует, пост считается calibration_lost и функция возвращает None.

    С 4+ маркерами используется полная проективная гомография (cv2.findHomography).
    С ровно 2 маркерами (типовой случай камеры side) — homography-совместимое представление
    подобия (масштаб + поворот + смещение), т.к. фронтальная плоскость камеры 2 не требует
    полной перспективной коррекции.
    """
    required_ids = set(known_marker_positions_mm)
    if not required_ids.issubset(detected_markers):
        return None

    ordered_ids = sorted(required_ids)
    src_points = np.array([detected_markers[i].mean(axis=0) for i in ordered_ids], dtype=np.float32)
    dst_points = np.array([known_marker_positions_mm[i] for i in ordered_ids], dtype=np.float32)

    if len(ordered_ids) >= 4:
        homography, _ = cv2.findHomography(src_points, dst_points)
        return homography

    if len(ordered_ids) == 2:
        transform, _ = cv2.estimateAffinePartial2D(
            src_points.reshape(-1, 1, 2), dst_points.reshape(-1, 1, 2)
        )
        if transform is None:
            return None
        return np.vstack([transform, [0.0, 0.0, 1.0]])

    return None


def pixels_to_mm(point_px: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    """Переводит точку в пикселях кадра в мм плоскости сцены через гомографию."""
    point = np.array([[point_px]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography)
    x, y = transformed[0, 0]
    return float(x), float(y)
