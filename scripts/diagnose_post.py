"""Почему пост не меряет: что видно на кадрах прямо сейчас.

Отвечает на два самых частых вопроса стенда: «пропал захват» и «груз не находится».

    python scripts/diagnose_post.py            # пост 1, кадры из работающего сервиса
    python scripts/diagnose_post.py --post 2 --save

Кадры берутся из MJPEG-стрима сервиса, а не с камер напрямую: устройства заняты воркером и
второй раз не откроются. Если сервис не запущен — ключ --device откроет камеры сам.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.aruco_homography import detect_markers  # noqa: E402
from src.store import PostStore  # noqa: E402

BORDER_SHARE = 0.08  # доля кадра по краям, которую считаем заведомо фоном
COLOURLESS_SATURATION = 60  # выше этого фон уже цветной


def grab_from_service(post_id: str, role: str, base_url: str) -> np.ndarray | None:
    """Первый целый JPEG из multipart-потока."""
    with urllib.request.urlopen(f"{base_url}/posts/{post_id}/stream/{role}", timeout=15) as response:
        buffer = b""
        while len(buffer) < 8_000_000:
            chunk = response.read(16384)
            if not chunk:
                break
            buffer += chunk
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2)
            if start >= 0 and end > start:
                return cv2.imdecode(np.frombuffer(buffer[start:end + 2], np.uint8), cv2.IMREAD_COLOR)
    return None


def grab_from_device(device_id: int, resolution: tuple[int, int]) -> np.ndarray | None:
    capture = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    try:
        for _ in range(5):  # первые кадры камера отдаёт до автоэкспозиции
            ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def suggest_background(hsv: np.ndarray) -> tuple[tuple, tuple, str]:
    """Пороги фона по краям кадра — там груза заведомо нет.

    Развилка принципиальная. Если фон бесцветный (стол, стена), объект отделяется только по
    насыщенности, и белый груз найтись не может в принципе: он такой же бесцветный, как фон.
    Если фон цветной (зелёный лист), объектом становится всё, что не этого цвета, — и белый груз
    находится наравне с остальными.
    """
    height, width = hsv.shape[:2]
    margin_y, margin_x = int(height * BORDER_SHARE), int(width * BORDER_SHARE)
    ring = np.vstack([
        hsv[:margin_y].reshape(-1, 3), hsv[-margin_y:].reshape(-1, 3),
        hsv[:, :margin_x].reshape(-1, 3), hsv[:, -margin_x:].reshape(-1, 3),
    ])

    saturation = ring[:, 1]
    if np.percentile(saturation, 90) < COLOURLESS_SATURATION:
        # Потолок берётся с запасом над фоном, но не до самого верха: в краевую полосу нередко
        # попадает и сам груз, и от 99-го процентиля порог уезжает туда, где объектом не
        # считается уже ничего.
        ceiling = int(min(np.percentile(saturation, 97) + 20, 160))
        return ((0, 0, 0), (180, ceiling, 255),
                "фон БЕСЦВЕТНЫЙ — груз отделяется только по насыщенности. "
                "Белый или серый груз так не найдётся никогда: он такой же бесцветный, как фон")

    hue = ring[:, 0]
    low, high = int(np.percentile(hue, 2)) - 8, int(np.percentile(hue, 98)) + 8
    return ((max(low, 0), 40, 30), (min(high, 180), 255, 255),
            "фон ЦВЕТНОЙ — объектом считается всё, что не этого цвета. Годится груз любого цвета, "
            "включая белый")


def report(role: str, frame: np.ndarray, binding) -> None:
    print(f"\n=== {role}: кадр {frame.shape[1]}x{frame.shape[0]}")

    found = set(detect_markers(frame))
    expected = set(binding.aruco_marker_ids)
    missing = sorted(expected - found)
    print(f"  маркеры: нужны {sorted(expected)}, видны {sorted(found)}")
    if missing:
        print(f"  ✗ НЕ ХВАТАЕТ {missing} — пост уходит в calibration_lost и не меряет вообще.")
        print("    Обычная причина: маркер закрыт грузом, ушёл из кадра или засвечен бликом.")
    else:
        print("  ✓ калибровка на месте")

    x, y, w, h = binding.roi
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    background = cv2.inRange(hsv, np.array(binding.background_hsv_lower),
                             np.array(binding.background_hsv_upper))
    object_mask = cv2.bitwise_not(background)
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max((cv2.contourArea(c) for c in contours), default=0.0)

    print(f"  порог фона сейчас: {list(binding.background_hsv_lower)} .. "
          f"{list(binding.background_hsv_upper)}")
    print(f"  объектом считается {100 - background.mean() / 2.55:.1f}% ROI, "
          f"крупнейший контур {largest:.0f} px при пороге {binding.min_contour_area_px}")
    if largest < binding.min_contour_area_px:
        print("  ✗ груз не найден: крупнейший контур меньше порога")

    lower, upper, note = suggest_background(hsv)
    print(f"  {note}")
    print(f"  порог по этому кадру: {list(lower)} .. {list(upper)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post", default="1")
    parser.add_argument("--url", default="http://127.0.0.1:8012")
    parser.add_argument("--device", action="store_true",
                        help="открыть камеры напрямую (сервис должен быть остановлен)")
    parser.add_argument("--save", action="store_true", help="сохранить кадры рядом со скриптом")
    args = parser.parse_args()

    post = PostStore("data/posts.json").get(args.post)
    if post is None:
        print(f"Точка {args.post} не найдена в data/posts.json")
        return 1

    for role, binding in (("top", post.camera_top), ("side", post.camera_side)):
        frame = (grab_from_device(binding.device_id, binding.resolution) if args.device
                 else grab_from_service(args.post, role, args.url))
        if frame is None:
            print(f"\n=== {role}: кадр не пришёл "
                  f"({'камера занята или не отвечает' if args.device else 'сервис не отдаёт поток'})")
            continue
        report(role, frame, binding)
        if args.save:
            cv2.imwrite(f"diagnose_{args.post}_{role}.jpg", frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
