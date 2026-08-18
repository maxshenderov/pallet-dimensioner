"""Усреднение и фильтрация измерений по скользящему окну кадров."""
from __future__ import annotations

import statistics
from collections import deque

from ..models import MeasurementResult


class MeasurementStabilizer:
    """Накапливает последние N измерений (L, W, H) и определяет момент их стабилизации."""

    def __init__(self, window_size: int, stddev_threshold_mm: float):
        self._window_size = window_size
        self._stddev_threshold_mm = stddev_threshold_mm
        self._lengths: deque[float] = deque(maxlen=window_size)
        self._widths: deque[float] = deque(maxlen=window_size)
        self._heights: deque[float] = deque(maxlen=window_size)

    def add_sample(self, l: float, w: float, h: float) -> None:
        self._lengths.append(l)
        self._widths.append(w)
        self._heights.append(h)

    def is_stable(self) -> bool:
        """True, если окно заполнено и std по каждому измерению не превышает порог."""
        if len(self._lengths) < self._window_size:
            return False
        return all(
            statistics.pstdev(dim) <= self._stddev_threshold_mm
            for dim in (self._lengths, self._widths, self._heights)
        )

    def get_stable_result(self) -> MeasurementResult | None:
        """Медианное значение по накопленному окну, если стабильно, иначе None."""
        if not self.is_stable():
            return None
        return MeasurementResult(
            length_mm=statistics.median(self._lengths),
            width_mm=statistics.median(self._widths),
            height_mm=statistics.median(self._heights),
            samples_count=len(self._lengths),
        )

    def reset(self) -> None:
        self._lengths.clear()
        self._widths.clear()
        self._heights.clear()
