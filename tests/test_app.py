"""Smoke-тесты FastAPI-приложения: CRUD точек, страницы, graceful camera_offline без реального железа."""
import time

import pytest
from fastapi.testclient import TestClient

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import app as app_module
    from src.store import PostStore
    from src.worker import PostWorkerRegistry

    app_module.registry.stop_all()
    app_module.store = PostStore(tmp_path / "posts.json")
    app_module.registry = PostWorkerRegistry()
    yield TestClient(app_module.app)
    app_module.registry.stop_all()


def _create_post(client, **overrides):
    body = {
        "name": "Тестовый пост",
        "camera_top_device_id": 97,
        "camera_side_device_id": 98,
        **overrides,
    }
    res = client.post("/api/posts", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_health():
    from fastapi.testclient import TestClient
    import app as app_module
    res = TestClient(app_module.app).get("/health")
    assert res.json() == {"status": "ok"}


def test_docs_and_posts_pages_render(client):
    assert client.get("/").status_code == 200
    assert client.get("/posts").status_code == 200


def test_create_list_get_post(client):
    created = _create_post(client)
    assert created["id"]
    assert created["camera_top"]["device_id"] == 97

    listed = client.get("/api/posts").json()
    assert len(listed) == 1

    fetched = client.get(f"/api/posts/{created['id']}").json()
    assert fetched["name"] == "Тестовый пост"


def test_get_missing_post_404(client):
    assert client.get("/api/posts/does-not-exist").status_code == 404


def test_workplace_page_renders_for_existing_post(client):
    created = _create_post(client)
    res = client.get(f"/posts/{created['id']}/workplace")
    assert res.status_code == 200
    assert created["id"] in res.text


def test_patch_updates_post(client):
    created = _create_post(client)
    res = client.patch(f"/api/posts/{created['id']}", json={"name": "Новое имя"})
    assert res.status_code == 200
    assert res.json()["name"] == "Новое имя"
    assert client.get(f"/api/posts/{created['id']}").json()["name"] == "Новое имя"


def test_start_without_real_camera_reports_camera_offline(client):
    created = _create_post(client)
    post_id = created["id"]

    res = client.post(f"/api/posts/{post_id}/start")
    assert res.status_code == 200

    state = None
    for _ in range(20):
        state_res = client.get(f"/api/posts/{post_id}/state")
        if state_res.status_code == 200:
            state = state_res.json()
            if state["status"] == "camera_offline":
                break
        time.sleep(0.2)

    assert state is not None
    assert state["status"] == "camera_offline"

    client.post(f"/api/posts/{post_id}/stop")


def test_stream_without_running_worker_returns_409(client):
    created = _create_post(client)
    res = client.get(f"/posts/{created['id']}/stream/top")
    assert res.status_code == 409


def test_qr_endpoint_works_before_any_measurement(client):
    created = _create_post(client)
    res = client.get(f"/posts/{created['id']}/qr.png")
    assert res.status_code == 200
    assert res.content.startswith(PNG_MAGIC)


def test_delete_post(client):
    created = _create_post(client)
    res = client.delete(f"/api/posts/{created['id']}")
    assert res.status_code == 204
    assert client.get(f"/api/posts/{created['id']}").status_code == 404
