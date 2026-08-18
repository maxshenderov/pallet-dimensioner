"""Выделение контура паллеты на кадре: ROI + HSV-порог фона + findContours."""
from __future__ import annotations

import cv2
import numpy as np

Contour = np.ndarray


def segment_pallet(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    background_hsv_lower: tuple[int, int, int],
    background_hsv_upper: tuple[int, int, int],
    min_contour_area_px: int,
) -> Contour | None:
    """
    Находит контур паллеты в ROI кадра.

    Возвращает контур с максимальной площадью (в координатах полного кадра) или None,
    если ничего не найдено (паллеты ещё нет на посту — штатная ситуация, не ошибка).
    Чистая функция без побочных эффектов — тестируется на фиксированных изображениях.
    """
    x, y, w, h = roi
    cropped = frame[y:y + h, x:x + w]

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    background_mask = cv2.inRange(hsv, np.array(background_hsv_lower), np.array(background_hsv_upper))
    object_mask = cv2.bitwise_not(background_mask)

    # open убирает мелкий шум (блики, зернистость), close закрывает мелкие дыры внутри
    # объекта — без этого контур дрожит от кадра к кадру даже при неподвижном объекте.
    kernel = np.ones((7, 7), np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_contour_area_px:
        return None

    return largest + np.array([x, y])
