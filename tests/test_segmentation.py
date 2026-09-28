"""Тесты vision/segmentation.py на синтетических изображениях (без физических камер)."""
import cv2
import numpy as np
import pytest

from src.vision.segmentation import _boxes_belong_together, segment_pallet

BACKGROUND_HSV_LOWER = (0, 0, 0)
BACKGROUND_HSV_UPPER = (180, 60, 90)
ROI = (0, 0, 400, 300)


def _frame_with_object(present: bool) -> np.ndarray:
    frame = np.full((300, 400, 3), 40, dtype=np.uint8)  # тёмный фон — попадает в HSV-диапазон фона
    if present:
        cv2.rectangle(frame, (100, 80), (300, 220), (200, 200, 200), thickness=-1)  # светлый объект
    return frame


def test_segment_pallet_finds_object():
    frame = _frame_with_object(present=True)
    contour = segment_pallet(frame, ROI, BACKGROUND_HSV_LOWER, BACKGROUND_HSV_UPPER, min_contour_area_px=1000)
    assert contour is not None
    _, _, w, h = cv2.boundingRect(contour)
    assert w == pytest.approx(200, abs=10)
    assert h == pytest.approx(140, abs=10)


def test_segment_pallet_no_object_returns_none():
    frame = _frame_with_object(present=False)
    contour = segment_pallet(frame, ROI, BACKGROUND_HSV_LOWER, BACKGROUND_HSV_UPPER, min_contour_area_px=1000)
    assert contour is None


def test_segment_pallet_filters_small_noise():
    frame = _frame_with_object(present=True)
    contour = segment_pallet(frame, ROI, BACKGROUND_HSV_LOWER, BACKGROUND_HSV_UPPER, min_contour_area_px=10_000_000)
    assert contour is None


class TestSplitCargo:
    """Груз, развалившийся в маске на куски, должен собираться обратно.

    Стеклянная банка на стенде: блик на стекле — чистый фон по порогу, он разрезал маску поперёк,
    крышка ушла отдельным куском, и высота считалась только по телу. Ошибка от такого разрыва
    всегда в одну сторону — вниз, и наружу никак не выходит.
    """

    def _jar(self, gap: int, lid: bool = True) -> np.ndarray:
        frame = np.full((300, 400, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (150, 120), (250, 220), (200, 200, 200), -1)  # тело
        if lid:
            cv2.rectangle(frame, (150, 60), (250, 120 - gap), (200, 200, 200), -1)  # крышка
        return frame

    def measure(self, frame):
        contour = segment_pallet(frame, ROI, BACKGROUND_HSV_LOWER, BACKGROUND_HSV_UPPER,
                                 min_contour_area_px=1000)
        return cv2.boundingRect(contour) if contour is not None else None

    def test_lid_separated_by_a_glare_band_is_taken_back(self):
        _, y, _, h = self.measure(self._jar(gap=6))
        assert y == pytest.approx(60, abs=6)
        assert h == pytest.approx(160, abs=10)

    def test_without_the_lid_the_height_is_the_body_alone(self):
        """Контрольный случай: без крышки высота обязана остаться прежней."""
        _, y, _, h = self.measure(self._jar(gap=6, lid=False))
        assert y == pytest.approx(120, abs=6)
        assert h == pytest.approx(100, abs=10)

    def test_a_neighbour_standing_apart_is_not_swallowed(self):
        """Посторонний предмет рядом — не кусок груза. На стенде это была чёрная стойка сбоку."""
        frame = self._jar(gap=6, lid=False)
        cv2.rectangle(frame, (20, 120), (90, 220), (200, 200, 200), -1)
        x, _, w, _ = self.measure(frame)
        assert x == pytest.approx(150, abs=6)
        assert w == pytest.approx(100, abs=10)

    def test_sideways_neighbours_never_join(self):
        """Склейка идёт только по вертикали — правило проверяется напрямую, без морфологии.

        Через кадр это не проверить: морфологическое закрытие 7x7 само сшивает всё, что ближе
        7 px, и тест мерил бы её, а не правило. Присоединение соседа СБОКУ на стенде цепляло к
        банке лежащую рядом бумагу, и видимая ширина прыгала с 58 до 96 мм.
        """
        main = (150, 120, 100, 100)
        above = (150, 20, 100, 96)  # ровно над грузом, разрыв 4 px
        beside = (140, 120, 100, 100)  # вплотную сбоку, полное перекрытие по вертикали

        assert _boxes_belong_together(main, above, gap=10)
        assert not _boxes_belong_together(main, beside, gap=10)

    def test_a_far_away_fragment_is_not_taken(self):
        frame = self._jar(gap=60)
        _, y, _, h = self.measure(frame)
        assert y == pytest.approx(120, abs=6)
        assert h == pytest.approx(100, abs=10)

    def test_merging_works_when_the_piece_is_not_the_first_found(self):
        """В кадре несколько посторонних кусков, и нужный — не первый по счёту.

        Куски раньше вычёркивались из списка через `list.remove`, а он сравнивает элементы
        оператором `==`; на контурах разной длины numpy отвечает исключением, и поток воркера
        падал целиком. В простом кадре это не всплывало: нужный кусок оказывался первым, и
        сравнение коротко замыкалось по тождеству.
        """
        frame = self._jar(gap=6)
        for left in (10, 55, 100):  # мелочь по краям, до груза не дотягивается
            cv2.rectangle(frame, (left, 250), (left + 35, 285), (200, 200, 200), -1)
        _, y, _, h = self.measure(frame)
        assert y == pytest.approx(60, abs=6)
        assert h == pytest.approx(160, abs=10)


def test_segment_pallet_contour_in_full_frame_coordinates():
    roi = (50, 30, 300, 250)
    frame = np.full((300, 400, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (150, 110), (350, 250), (200, 200, 200), thickness=-1)

    contour = segment_pallet(frame, roi, BACKGROUND_HSV_LOWER, BACKGROUND_HSV_UPPER, min_contour_area_px=1000)
    assert contour is not None
    x, y, w, h = cv2.boundingRect(contour)
    assert x == pytest.approx(150, abs=5)
    assert y == pytest.approx(110, abs=5)
