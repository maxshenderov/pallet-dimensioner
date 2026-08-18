"""Генерация QR-кода с текущими данными измерения точки (габариты + вес)."""
from __future__ import annotations

import io
import json

import qrcode

from .models import Post, PostState


def build_payload(post: Post, state: PostState) -> dict:
    """Компактный JSON, зашиваемый в QR — то, что клеится на паллету / читается сканером."""
    payload = {
        "post_id": post.id,
        "post_name": post.name,
    }
    if state.length_mm is not None:
        payload["length_mm"] = round(state.length_mm)
    if state.width_mm is not None:
        payload["width_mm"] = round(state.width_mm)
    if state.height_mm is not None:
        payload["height_mm"] = round(state.height_mm)
    if state.cross_check_passed is not None:
        payload["cross_check_passed"] = state.cross_check_passed
    if state.weight and state.weight.ok and state.weight.value is not None:
        payload["weight"] = state.weight.value
        payload["weight_unit"] = state.weight.unit
    if state.updated_at is not None:
        payload["timestamp"] = state.updated_at.isoformat()
    return payload


def generate_png(post: Post, state: PostState) -> bytes:
    """PNG-байты QR-кода с данными измерения. Если данных ещё нет — кодирует только id поста."""
    payload = build_payload(post, state)
    qr = qrcode.QRCode(border=2, box_size=8)
    qr.add_data(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
