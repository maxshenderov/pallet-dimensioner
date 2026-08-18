"""JSON-файловое хранилище точек измерения (posts) — без внешней БД, как и остальные сервисы проекта."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import Post


def _slugify(name: str, existing_ids: set[str]) -> str:
    """ASCII-only slug для id точки — используется в URL, curl-примерах и QR, поэтому
    не-ASCII имя (например, кириллица) не транслитерируется, а заменяется коротким id."""
    ascii_chars = "".join(c.lower() if c.isalnum() and c.isascii() else "-" for c in name)
    base = ascii_chars.strip("-") or f"post-{uuid.uuid4().hex[:6]}"
    slug = base
    suffix = 1
    while slug in existing_ids:
        suffix += 1
        slug = f"{base}-{suffix}"
    return slug


class PostStore:
    """Простое потокобезопасное CRUD-хранилище точек в одном JSON-файле."""

    def __init__(self, path: str | Path = "data/posts.json"):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({})

    def _read(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8") or "{}")

    def _write(self, data: dict[str, dict]) -> None:
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list(self) -> list[Post]:
        with self._lock:
            return [Post(**p) for p in self._read().values()]

    def get(self, post_id: str) -> Post | None:
        with self._lock:
            raw = self._read().get(post_id)
            return Post(**raw) if raw else None

    def create(self, post: Post) -> Post:
        with self._lock:
            data = self._read()
            if not post.id:
                post = post.model_copy(update={"id": _slugify(post.name, set(data))})
            data[post.id] = json.loads(post.model_dump_json())
            self._write(data)
            return post

    def update(self, post_id: str, updated: Post) -> Post:
        with self._lock:
            data = self._read()
            if post_id not in data:
                raise KeyError(post_id)
            data[post_id] = json.loads(updated.model_dump_json())
            self._write(data)
            return updated

    def delete(self, post_id: str) -> None:
        with self._lock:
            data = self._read()
            data.pop(post_id, None)
            self._write(data)


def new_post_id(name: str, store: PostStore) -> str:
    existing = {p.id for p in store.list()}
    return _slugify(name, existing)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid_suffix() -> str:
    return uuid.uuid4().hex[:6]
