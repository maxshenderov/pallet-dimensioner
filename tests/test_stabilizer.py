"""Тесты vision/stabilizer.py на синтетических сериях сэмплов (стабильные/нестабильные случаи)."""
import pytest

from src.vision.stabilizer import MeasurementStabilizer


def test_not_stable_until_window_full():
    stabilizer = MeasurementStabilizer(window_size=5, stddev_threshold_mm=10)
    for _ in range(4):
        stabilizer.add_sample(1000, 800, 1400)
    assert stabilizer.is_stable() is False
    assert stabilizer.get_stable_result() is None


def test_stable_when_samples_consistent():
    stabilizer = MeasurementStabilizer(window_size=5, stddev_threshold_mm=10)
    samples = [(1000, 800, 1400), (1002, 799, 1401), (998, 801, 1399), (1001, 800, 1400), (999, 800, 1401)]
    for l, w, h in samples:
        stabilizer.add_sample(l, w, h)

    assert stabilizer.is_stable() is True
    result = stabilizer.get_stable_result()
    assert result is not None
    assert result.length_mm == pytest.approx(1000, abs=5)
    assert result.samples_count == 5


def test_unstable_when_samples_noisy():
    stabilizer = MeasurementStabilizer(window_size=5, stddev_threshold_mm=5)
    for l in (1000, 1050, 950, 1100, 900):
        stabilizer.add_sample(l, 800, 1400)
    assert stabilizer.is_stable() is False
    assert stabilizer.get_stable_result() is None


def test_reset_clears_window():
    stabilizer = MeasurementStabilizer(window_size=3, stddev_threshold_mm=10)
    for _ in range(3):
        stabilizer.add_sample(1000, 800, 1400)
    assert stabilizer.is_stable() is True

    stabilizer.reset()
    assert stabilizer.is_stable() is False
