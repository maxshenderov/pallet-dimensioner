"""Выделение контура паллеты на кадре: ROI + HSV-порог фона + findContours."""
from __future__ import annotations

import cv2
import numpy as np

Contour = np.ndarray


MARKER_MARGIN_SHARE = 0.06  # запас вокруг метки: детектор отдаёт углы чёрной рамки без полей


def _erase_quads(mask: np.ndarray, quads, offset: tuple[int, int]) -> None:
    """Стирает из маски объекта четырёхугольники калибровочных меток (на месте).

    Чёрные поля ArUco-метки фоном не считаются: тон у почти чёрного пикселя шумит и частью
    уходит ниже нижней границы. На стенде маска отдавала под «объект» 24 % площади метки
    215x215 px — 11 тыс. px при пороге контура 3000. Пока метка стоит в стороне, побеждает
    контур груза; стоит ей оказаться рядом — морфологическое закрытие сшивает её с грузом в
    один контур, и габарит уезжает на ширину метки. Метки мы и так находим на каждом кадре
    ради калибровки, поэтому дешевле всего вычесть их до сегментации.

    Стирается только НАЙДЕННАЯ метка. Закрытую грузом метку детектор не отдаёт — и хорошо:
    иначе вместе с ней стёрся бы кусок самого груза.
    """
    for corners in quads:
        quad = np.asarray(corners, dtype=np.float32).reshape(-1, 2) - np.asarray(offset, np.float32)
        centre = quad.mean(axis=0)
        margin = 1.0 + MARKER_MARGIN_SHARE
        cv2.fillConvexPoly(mask, np.int32(centre + (quad - centre) * margin), 0)


MERGE_GAP_SHARE = 0.05  # разрыв между кусками одного груза — доля размера крупнейшего куска
MERGE_GAP_MIN_PX = 8
MERGE_OVERLAP_SHARE = 0.5  # насколько кусок должен попадать в полосу крупнейшего, чтобы быть им


def _boxes_belong_together(main, other, gap: float) -> bool:
    """Кусок принадлежит грузу, если стоит РОВНО НАД или ПОД основным и вплотную к нему.

    Только по вертикали, и это не упрощение. Наблюдаемые разрывы маски идут поперёк груза:
    блик на стекле, стяжная лента, шов плёнки — все они кладут горизонтальную полосу фона.
    Куски при этом стоят друг над другом. Присоединение соседа СБОКУ ничем не подтверждено, а
    вред от него измерен: на стенде оно цепляло к банке бумагу рядом, и видимая ширина прыгала
    с 58 до 96 мм. Понадобится (вертикальный разрыв от ленты) — вернём по факту, не раньше.

    Условий два, и второе не лишнее. Кусок должен попадать в вертикальную полосу основного
    больше чем наполовину — иначе прилипнет сосед по диагонали. И он должен лежать в основном
    ВНЕ его высоты: сосед, стоящий вплотную сбоку, перекрывается по горизонтали ничуть не хуже
    крышки, и отличает их только то, что крышка стоит выше груза, а сосед — на той же высоте.
    """
    ax, ay, aw, ah = main
    bx, by, bw, bh = other
    overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
    overlap_y = min(ay + ah, by + bh) - max(ay, by)
    gap_x = max(ax - (bx + bw), bx - (ax + aw), 0)
    gap_y = max(ay - (by + bh), by - (ay + ah), 0)

    return (overlap_x >= MERGE_OVERLAP_SHARE * bw
            and overlap_y <= MERGE_OVERLAP_SHARE * bh
            and gap_y <= gap and gap_x <= 0)


def _merge_fragments(largest: Contour, contours) -> Contour:
    """Собирает груз, развалившийся в маске на несколько кусков.

    Раньше возвращался ОДИН крупнейший контур, и всё остальное молча пропадало. На стенде
    стеклянная банка с бордовой крышкой развалилась надвое: блик на стекле (тон 104, насыщенность
    19 — то есть чистый фон) разрезал маску поперёк. Крышка ушла отдельным куском 348x144, тело
    осталось куском 332x263 с разрывом в 4 px, и высота считалась только по телу — занижение на
    треть. На настоящей паллете так же рвут маску стрейч-плёнка, стяжные ленты и швы упаковки,
    и ошибка всегда в одну сторону — вниз.

    Куски собираются вокруг крупнейшего и только те, что стоят с ним в одной полосе вплотную
    (`_boxes_belong_together`). Присоединение идёт до тех пор, пока находятся новые: крышка может
    цепляться не к телу напрямую, а через кусок между ними.

    Результат — выпуклая оболочка объединения. Склеенный список точек полигоном не является, и
    `contourArea` на нём даёт бессмыслицу; оболочка же остаётся корректным контуром. Когда кусок
    один — возвращается он сам, без всякой оболочки, то есть прежнее поведение не меняется.
    """
    box = cv2.boundingRect(largest)
    gap = max(MERGE_GAP_MIN_PX, MERGE_GAP_SHARE * max(box[2], box[3]))

    # Куски отслеживаются НОМЕРАМИ, а не самими массивами: list.remove сравнивает элементы
    # через `==`, а на контурах разной длины numpy отвечает на это исключением.
    taken = {i for i, c in enumerate(contours) if c is largest}
    kept = [largest]
    growing = True
    while growing:
        growing = False
        for i, contour in enumerate(contours):
            if i in taken or not _boxes_belong_together(box, cv2.boundingRect(contour), gap):
                continue
            taken.add(i)
            kept.append(contour)
            box = cv2.boundingRect(np.vstack(kept))
            growing = True

    if len(kept) == 1:
        return largest
    return cv2.convexHull(np.vstack(kept))


def segment_pallet(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    background_hsv_lower: tuple[int, int, int],
    background_hsv_upper: tuple[int, int, int],
    min_contour_area_px: int,
    marker_quads=(),
) -> Contour | None:
    """
    Находит контур паллеты в ROI кадра.

    Возвращает контур с максимальной площадью (в координатах полного кадра) или None,
    если ничего не найдено (паллеты ещё нет на посту — штатная ситуация, не ошибка).
    Чистая функция без побочных эффектов — тестируется на фиксированных изображениях.

    `marker_quads` — углы калибровочных меток в координатах полного кадра (то, что отдаёт
    `detect_markers`). Они вычитаются из маски: см. `_erase_quads`.
    """
    x, y, w, h = roi
    cropped = frame[y:y + h, x:x + w]

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    background_mask = cv2.inRange(hsv, np.array(background_hsv_lower), np.array(background_hsv_upper))
    object_mask = cv2.bitwise_not(background_mask)
    _erase_quads(object_mask, marker_quads, (x, y))

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

    return _merge_fragments(largest, contours) + np.array([x, y])
