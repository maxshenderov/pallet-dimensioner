#!/usr/bin/env python
"""Интерактивная калибровка поста: подбор ROI и HSV-порогов фона, детекция ArUco, сохранение в конфиг."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import yaml

from src.calibration.aruco_homography import detect_markers

WINDOW = "calibrate_post"
TRACKBAR_KEYS = ("roi_x", "roi_y", "roi_w", "roi_h", "h_lo", "s_lo", "v_lo", "h_hi", "s_hi", "v_hi")


def _nothing(_value: int) -> None:
    pass


def _create_trackbars(cam_config: dict) -> None:
    h, w = cam_config["resolution"][1], cam_config["resolution"][0]
    roi = cam_config["roi"]
    hsv_lower = cam_config["background_hsv_lower"]
    hsv_upper = cam_config["background_hsv_upper"]

    cv2.createTrackbar("roi_x", WINDOW, roi[0], w, _nothing)
    cv2.createTrackbar("roi_y", WINDOW, roi[1], h, _nothing)
    cv2.createTrackbar("roi_w", WINDOW, roi[2], w, _nothing)
    cv2.createTrackbar("roi_h", WINDOW, roi[3], h, _nothing)
    cv2.createTrackbar("h_lo", WINDOW, hsv_lower[0], 180, _nothing)
    cv2.createTrackbar("s_lo", WINDOW, hsv_lower[1], 255, _nothing)
    cv2.createTrackbar("v_lo", WINDOW, hsv_lower[2], 255, _nothing)
    cv2.createTrackbar("h_hi", WINDOW, hsv_upper[0], 180, _nothing)
    cv2.createTrackbar("s_hi", WINDOW, hsv_upper[1], 255, _nothing)
    cv2.createTrackbar("v_hi", WINDOW, hsv_upper[2], 255, _nothing)


def _read_trackbars() -> tuple[list[int], list[int], list[int]]:
    values = {k: cv2.getTrackbarPos(k, WINDOW) for k in TRACKBAR_KEYS}
    roi = [values["roi_x"], values["roi_y"], values["roi_w"], values["roi_h"]]
    hsv_lower = [values["h_lo"], values["s_lo"], values["v_lo"]]
    hsv_upper = [values["h_hi"], values["s_hi"], values["v_hi"]]
    return roi, hsv_lower, hsv_upper


def run(camera_key: str, config_path: str) -> None:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    cam_config = config[f"camera_{camera_key}"]

    cap = cv2.VideoCapture(cam_config["device_id"])
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {cam_config['device_id']}")

    cv2.namedWindow(WINDOW)
    _create_trackbars(cam_config)

    print("Управление: 'a' — детекция ArUco в текущем кадре, 's' — сохранить ROI/HSV в конфиг, 'q' — выход")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Кадр не получен, повтор...")
                continue

            roi, hsv_lower, hsv_upper = _read_trackbars()
            x, y, w, h = roi
            preview = frame.copy()
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(WINDOW, preview)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("a"):
                markers = detect_markers(frame)
                print(f"Найдены маркеры: {sorted(markers)}")
            elif key == ord("s"):
                cam_config["roi"] = roi
                cam_config["background_hsv_lower"] = hsv_lower
                cam_config["background_hsv_upper"] = hsv_upper
                config_file.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
                print(f"Сохранено в {config_file}")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Интерактивная калибровка поста pallet-dimensioner")
    parser.add_argument("camera", choices=["top", "side"], help="Какую камеру калибровать")
    parser.add_argument("--config", default="config/post_config.yaml")
    args = parser.parse_args()
    run(args.camera, args.config)
