"""Захват кадров с двух камер, синхронизированный по минимальному временному разрыву."""
from __future__ import annotations

import cv2
import numpy as np


class CameraUnavailableError(Exception):
    """Камера не отвечает или не удаётся прочитать кадр (временный сбой USB и т.п.)."""


class CameraStream:
    """Обёртка над cv2.VideoCapture."""

    def __init__(self, camera_id: int, resolution: tuple[int, int]):
        self.camera_id = camera_id
        self._cap = cv2.VideoCapture(camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if not self._cap.isOpened():
            raise CameraUnavailableError(f"Не удалось открыть камеру {camera_id}")

    def get_frame(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraUnavailableError(f"Не удалось прочитать кадр с камеры {self.camera_id}")
        return frame

    def release(self) -> None:
        self._cap.release()


def get_synchronized_frames(stream1: CameraStream, stream2: CameraStream) -> tuple[np.ndarray, np.ndarray]:
    """
    Захват кадров с обеих камер с минимальным временным разрывом: сначала grab() на обеих
    камерах подряд (сброс внутреннего буфера), затем retrieve() — без промежуточного decode.
    """
    stream1._cap.grab()
    stream2._cap.grab()
    ok1, frame1 = stream1._cap.retrieve()
    ok2, frame2 = stream2._cap.retrieve()

    if not ok1 or frame1 is None:
        raise CameraUnavailableError(f"Не удалось прочитать кадр с камеры {stream1.camera_id}")
    if not ok2 or frame2 is None:
        raise CameraUnavailableError(f"Не удалось прочитать кадр с камеры {stream2.camera_id}")

    return frame1, frame2
