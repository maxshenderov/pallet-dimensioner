"""Тесты qr.py: сборка payload и генерация PNG, без камер и сети."""
from datetime import datetime, timezone

from src.models import CameraSideBinding, CameraTopBinding, Post, PostState, PostStatus, WeightReading
from src.qr import build_payload, generate_png

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _post() -> Post:
    return Post(
        id="post-04", name="Пост 04",
        camera_top=CameraTopBinding(device_id=0), camera_side=CameraSideBinding(device_id=1),
        created_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
    )


def test_build_payload_includes_only_available_fields():
    state = PostState(post_id="post-04", status=PostStatus.running)
    payload = build_payload(_post(), state)

    assert payload["post_id"] == "post-04"
    assert "height_mm" not in payload
    assert "weight" not in payload


def test_build_payload_includes_full_measurement():
    state = PostState(
        post_id="post-04", status=PostStatus.running, stable=True,
        height_mm=1487.0, width_mm=1195.4, depth_mm=800.1, cross_check_passed=True,
        weight=WeightReading(ok=True, value=820.5, unit="kg", stable=True),
        updated_at=datetime(2026, 8, 18, 10, 23, 41, tzinfo=timezone.utc),
    )
    payload = build_payload(_post(), state)

    assert payload["height_mm"] == 1487
    assert payload["width_mm"] == 1195
    assert payload["depth_mm"] == 800
    assert payload["cross_check_passed"] is True
    assert payload["weight"] == 820.5
    assert payload["weight_unit"] == "kg"
    assert payload["timestamp"].startswith("2026-08-18T10:23:41")


def test_build_payload_skips_weight_when_scale_offline():
    state = PostState(post_id="post-04", weight=WeightReading(ok=False))
    payload = build_payload(_post(), state)
    assert "weight" not in payload


def test_generate_png_returns_valid_png_bytes():
    state = PostState(post_id="post-04", height_mm=1450, width_mm=1200, depth_mm=800)
    png = generate_png(_post(), state)

    assert png.startswith(PNG_MAGIC)
    assert len(png) > 100


def test_generate_png_with_empty_state_still_works():
    png = generate_png(_post(), PostState(post_id="post-04"))
    assert png.startswith(PNG_MAGIC)
