"""Тесты calibration/aruco_homography.py на синтетических кадрах с нарисованными ArUco-маркерами."""
import cv2
import numpy as np
import pytest

from src.calibration.aruco_homography import compute_homography, detect_markers, pixels_to_mm

DICTIONARY = "DICT_5X5_100"
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
MARKER_SIZE_PX = 80

# id -> (центр маркера в пикселях, соответствующая точка в мм)
MARKER_LAYOUT = {
    1: ((150, 150), (0.0, 0.0)),
    2: ((650, 150), (1200.0, 0.0)),
    3: ((650, 450), (1200.0, 800.0)),
    4: ((150, 450), (0.0, 800.0)),
}


def _synthetic_frame(marker_ids: list[int]) -> np.ndarray:
    canvas = np.full((600, 800, 3), 255, dtype=np.uint8)
    half = MARKER_SIZE_PX // 2
    for marker_id in marker_ids:
        (cx, cy), _ = MARKER_LAYOUT[marker_id]
        marker_img = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, MARKER_SIZE_PX)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        canvas[cy - half:cy - half + MARKER_SIZE_PX, cx - half:cx - half + MARKER_SIZE_PX] = marker_bgr
    return canvas


def test_detect_markers_finds_all_ids():
    frame = _synthetic_frame([1, 2, 3, 4])
    detected = detect_markers(frame, DICTIONARY)
    assert set(detected) == {1, 2, 3, 4}


def test_detect_markers_empty_frame_returns_empty_dict():
    frame = np.full((600, 800, 3), 255, dtype=np.uint8)
    assert detect_markers(frame, DICTIONARY) == {}


def test_compute_homography_full_set_maps_centers_to_known_mm():
    frame = _synthetic_frame([1, 2, 3, 4])
    detected = detect_markers(frame, DICTIONARY)
    known_positions = {mid: mm for mid, (_, mm) in MARKER_LAYOUT.items()}

    homography = compute_homography(detected, known_positions)
    assert homography is not None

    for marker_id, (px_center, mm_expected) in MARKER_LAYOUT.items():
        mm_actual = pixels_to_mm(px_center, homography)
        assert mm_actual[0] == pytest.approx(mm_expected[0], abs=15)
        assert mm_actual[1] == pytest.approx(mm_expected[1], abs=15)


def test_compute_homography_missing_marker_returns_none():
    frame = _synthetic_frame([1, 2, 3])  # маркер 4 отсутствует в кадре
    detected = detect_markers(frame, DICTIONARY)
    known_positions = {mid: mm for mid, (_, mm) in MARKER_LAYOUT.items()}

    assert compute_homography(detected, known_positions) is None


def test_compute_homography_two_markers_similarity_transform():
    known_positions = {1: (0.0, 0.0), 2: (1000.0, 0.0)}
    frame = _synthetic_frame([1, 2])
    detected = detect_markers(frame, DICTIONARY)

    homography = compute_homography(detected, known_positions)
    assert homography is not None

    mm1 = pixels_to_mm(MARKER_LAYOUT[1][0], homography)
    mm2 = pixels_to_mm(MARKER_LAYOUT[2][0], homography)
    assert mm1[0] == pytest.approx(0.0, abs=15)
    assert mm2[0] == pytest.approx(1000.0, abs=15)
