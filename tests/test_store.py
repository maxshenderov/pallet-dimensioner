"""Тесты store.py: CRUD точек в JSON-файле, без камер и сети."""
from datetime import datetime, timezone

from src.models import CameraSideBinding, CameraTopBinding, Post
from src.store import PostStore, new_post_id


def _post(name="Post 1", **overrides) -> Post:
    defaults = dict(
        id="",
        name=name,
        camera_top=CameraTopBinding(device_id=0),
        camera_side=CameraSideBinding(device_id=1),
        created_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return Post(**defaults)


def test_create_generates_slug_id(tmp_path):
    store = PostStore(tmp_path / "posts.json")
    created = store.create(_post("Post 04"))
    assert created.id == "post-04"


def test_create_with_non_ascii_name_falls_back_to_short_id(tmp_path):
    """Кириллица (и любой не-ASCII) не транслитерируется — id должен остаться ASCII (URL/curl/QR)."""
    store = PostStore(tmp_path / "posts.json")
    created = store.create(_post("Пост"))
    assert created.id.isascii()
    assert created.id.startswith("post-")
    assert len(created.id) > len("post-")


def test_create_with_partial_ascii_name_keeps_ascii_part(tmp_path):
    """Если в имени есть ASCII-фрагмент (например, номер) — он используется как есть."""
    store = PostStore(tmp_path / "posts.json")
    created = store.create(_post("Пост 04"))
    assert created.id == "04"


def test_create_dedupes_slug_on_collision(tmp_path):
    store = PostStore(tmp_path / "posts.json")
    store.create(_post("Post"))
    second = store.create(_post("Post"))
    assert second.id == "post-2"


def test_list_and_get_roundtrip(tmp_path):
    store = PostStore(tmp_path / "posts.json")
    created = store.create(_post("A"))

    assert [p.id for p in store.list()] == [created.id]
    fetched = store.get(created.id)
    assert fetched is not None
    assert fetched.camera_top.device_id == 0


def test_get_missing_returns_none(tmp_path):
    store = PostStore(tmp_path / "posts.json")
    assert store.get("nope") is None


def test_update_persists_changes(tmp_path):
    store = PostStore(tmp_path / "posts.json")
    created = store.create(_post("A"))

    updated = created.model_copy(update={"name": "B"})
    store.update(created.id, updated)

    assert store.get(created.id).name == "B"


def test_delete_removes_post(tmp_path):
    store = PostStore(tmp_path / "posts.json")
    created = store.create(_post("A"))

    store.delete(created.id)

    assert store.get(created.id) is None
    assert store.list() == []


def test_new_post_id_avoids_existing(tmp_path):
    store = PostStore(tmp_path / "posts.json")
    store.create(_post("Alpha", id="alpha"))
    assert new_post_id("Alpha", store) == "alpha-2"
