"""
Тесты calibration/camera_calibrate.py на синтетических кадрах шахматной доски.

Изображения генерируются программно: плоская доска рендерится и проецируется через
разные гомографии (что физически эквивалентно фото одной и той же плоской доски с
разных ракурсов при отсутствии дисторсии) — независимо от реальных камер, per ТЗ п.9.
"""
import cv2
import numpy as np
import pytest

from src.calibration.camera_calibrate import calibrate_lens, undistort

SQUARE_PX = 60
COLS, ROWS = 10, 7  # -> 9x6 внутренних углов
CANVAS_SIZE = (900, 700)  # (width, height)

BOARD_TRANSFORMS = [
    {"scale": 1.0, "shear": 0.0, "angle": 0, "tx": 60, "ty": 40},
    {"scale": 0.9, "shear": 0.06, "angle": 5, "tx": 40, "ty": 60},
    {"scale": 1.05, "shear": -0.05, "angle": -6, "tx": 70, "ty": 20},
    {"scale": 0.85, "shear": 0.08, "angle": 9, "tx": 20, "ty": 30},
    {"scale": 1.0, "shear": -0.08, "angle": -9, "tx": 40, "ty": 10},
    {"scale": 0.95, "shear": 0.04, "angle": 3, "tx": 55, "ty": 45},
]


def _board_image() -> np.ndarray:
    board = np.zeros((ROWS * SQUARE_PX, COLS * SQUARE_PX), dtype=np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 0:
                board[r * SQUARE_PX:(r + 1) * SQUARE_PX, c * SQUARE_PX:(c + 1) * SQUARE_PX] = 255
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def _warp(board: np.ndarray, params: dict) -> np.ndarray:
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    angle = np.radians(params["angle"])
    scale = params["scale"]
    shear = params["shear"]
    matrix2x2 = np.array(
        [[scale * np.cos(angle), -np.sin(angle) + shear], [np.sin(angle), scale * np.cos(angle)]]
    )
    dst = (src - [w / 2, h / 2]) @ matrix2x2.T + [w / 2 + params["tx"], h / 2 + params["ty"]]

    homography = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), 255, dtype=np.uint8)
    return cv2.warpPerspective(
        board, homography, CANVAS_SIZE, dst=canvas, borderMode=cv2.BORDER_TRANSPARENT
    )


@pytest.fixture
def chessboard_image_paths(tmp_path):
    board = _board_image()
    paths = []
    for i, params in enumerate(BOARD_TRANSFORMS):
        frame = _warp(board, params)
        path = tmp_path / f"chessboard_{i}.png"
        cv2.imwrite(str(path), frame)
        paths.append(str(path))
    return paths


def test_calibrate_lens_returns_plausible_intrinsics(chessboard_image_paths):
    intrinsics = calibrate_lens(
        chessboard_image_paths, chessboard_size=(9, 6), square_size_mm=25.0, camera_id="test"
    )

    assert intrinsics.camera_matrix[0][0] > 0  # fx
    assert intrinsics.camera_matrix[1][1] > 0  # fy
    assert intrinsics.image_size == CANVAS_SIZE
    assert intrinsics.reprojection_error < 5.0


def test_calibrate_lens_too_few_valid_images_raises(chessboard_image_paths):
    with pytest.raises(ValueError):
        calibrate_lens(chessboard_image_paths[:2], chessboard_size=(9, 6))


def test_calibrate_lens_unreadable_image_raises(chessboard_image_paths, tmp_path):
    bad_path = tmp_path / "not_an_image.png"
    bad_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError):
        calibrate_lens([*chessboard_image_paths, str(bad_path)], chessboard_size=(9, 6))


def test_undistort_preserves_shape(chessboard_image_paths):
    intrinsics = calibrate_lens(
        chessboard_image_paths, chessboard_size=(9, 6), square_size_mm=25.0, camera_id="test"
    )
    frame = cv2.imread(chessboard_image_paths[0])

    result = undistort(frame, intrinsics)

    assert result.shape == frame.shape
