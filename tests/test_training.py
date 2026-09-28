"""Тесты хранилища замеров, размеченной выборки и API обучения поправки."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.models import TrainingSample
from src.storage import Database
from src.training import TrainingStore
from src.vision.correction import BACKEND_HULL, BACKEND_ORTHO, MIN_SAMPLES


def make_sample(index: int, backend: str = BACKEND_ORTHO) -> TrainingSample:
    """Замер с систематическим завышением на 5 %, которое поправка и должна убрать."""
    height, width, depth = 400.0 + index * 20, 900.0 + index * 10, 600.0 + index * 5
    features = [height * 1.05, width * 1.05, depth * 1.05, width * 1.04,
                1500.0 + index, index * 7.0, 0.92, width / depth, width * 1.06]
    return TrainingSample(
        features=features, height_mm=height, width_mm=width, depth_mm=depth,
        backend=backend, created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "pallet.db")
    yield database
    database.close()


@pytest.fixture
def store(db):
    return TrainingStore(db, "p1")


class TestStore:
    def test_starts_empty_and_accumulates(self, store):
        assert store.list() == []
        assert store.add(make_sample(0)) == 1
        assert store.add(make_sample(1)) == 2
        assert len(store.list()) == 2

    def test_survives_a_reopen(self, tmp_path):
        """Ради этого база и заведена: данные переживают перезапуск сервиса и контейнера."""
        first = Database(tmp_path / "pallet.db")
        TrainingStore(first, "p1").add(make_sample(0))
        first.close()

        second = Database(tmp_path / "pallet.db")
        assert len(TrainingStore(second, "p1").list()) == 1
        second.close()

    def test_posts_do_not_see_each_others_samples(self, db):
        TrainingStore(db, "p1").add(make_sample(0))
        assert TrainingStore(db, "p2").list() == []

    def test_clear_removes_samples_and_model(self, store):
        for i in range(MIN_SAMPLES):
            store.add(make_sample(i))
        store.train(BACKEND_ORTHO)
        assert store.load_model() is not None

        store.clear()
        assert store.list() == []
        assert store.load_model() is None

    def test_refuses_to_train_on_too_few_samples(self, store):
        for i in range(MIN_SAMPLES - 1):
            store.add(make_sample(i))
        with pytest.raises(ValueError, match=str(MIN_SAMPLES)):
            store.train(BACKEND_ORTHO)

    def test_counts_only_samples_of_the_asked_backend(self, store):
        """Признаки резки и признаки метода передней грани несовместимы — смешивать нельзя."""
        for i in range(MIN_SAMPLES):
            store.add(make_sample(i, BACKEND_HULL))
        assert store.status(BACKEND_ORTHO).samples_count == 0
        with pytest.raises(ValueError):
            store.train(BACKEND_ORTHO)

    def test_training_reports_expected_error(self, store):
        for i in range(40):
            store.add(make_sample(i))
        status = store.train(BACKEND_ORTHO)

        assert status.trained is True
        assert status.trained_on_samples == 40
        assert status.expected_error_mm is not None
        # Завышение здесь ровно на 5 %, регрессия обязана снять его почти полностью.
        assert max(status.expected_error_mm) < 20.0

    def test_expected_error_outlives_a_restart(self, tmp_path):
        """Раньше оценка качества жила только в ответе на обучение и пропадала после
        перезапуска: модель работает, а насколько ей верить — уже не видно."""
        first = Database(tmp_path / "pallet.db")
        store = TrainingStore(first, "p1")
        for i in range(40):
            store.add(make_sample(i))
        expected = store.train(BACKEND_ORTHO).expected_error_mm
        first.close()

        second = Database(tmp_path / "pallet.db")
        assert TrainingStore(second, "p1").status(BACKEND_ORTHO).expected_error_mm == expected
        second.close()

    def test_model_is_reusable_after_reopen(self, db):
        store = TrainingStore(db, "p1")
        for i in range(MIN_SAMPLES):
            store.add(make_sample(i))
        store.train(BACKEND_ORTHO)

        model = TrainingStore(db, "p1").load_model()
        assert model is not None and model.backend == BACKEND_ORTHO
        assert model.clamp_to_raw is False  # метод передней грани ошибается в обе стороны

    def test_retraining_keeps_the_previous_model(self, db):
        """Старая модель — единственное, с чем сравнить новую, если габариты вдруг поехали."""
        store = TrainingStore(db, "p1")
        for i in range(MIN_SAMPLES):
            store.add(make_sample(i))
        store.train(BACKEND_ORTHO)
        store.add(make_sample(MIN_SAMPLES))
        store.train(BACKEND_ORTHO)

        rows = db._query("SELECT samples_count FROM correction_models WHERE post_id = 'p1' ORDER BY id")
        assert [row["samples_count"] for row in rows] == [MIN_SAMPLES, MIN_SAMPLES + 1]


class TestMeasurementJournal:
    def test_records_and_reads_back(self, db):
        db.add_measurement("p1", BACKEND_ORTHO, 100.0, 200.0, 300.0,
                           raw=(105.0, 210.0, 315.0), features=[1.0] * 9,
                           weight_kg=12.5, cross_check_passed=True, correction_applied=True)
        items = db.measurements("p1")

        assert db.measurements_count("p1") == 1
        assert items[0]["height_mm"] == 100.0
        assert items[0]["raw_height_mm"] == 105.0
        assert items[0]["features"] == [1.0] * 9
        assert items[0]["cross_check_passed"] is True
        assert items[0]["correction_applied"] is True

    def test_newest_first_and_paged(self, db):
        for i in range(5):
            db.add_measurement("p1", BACKEND_ORTHO, float(i), 1.0, 1.0,
                               raw=(0.0, 0.0, 0.0), features=[])
        assert [m["height_mm"] for m in db.measurements("p1", limit=2)] == [4.0, 3.0]
        assert [m["height_mm"] for m in db.measurements("p1", limit=2, offset=2)] == [2.0, 1.0]

    def test_posts_are_separate(self, db):
        db.add_measurement("p1", BACKEND_ORTHO, 1.0, 1.0, 1.0, raw=(0.0, 0.0, 0.0), features=[])
        assert db.measurements_count("p2") == 0


class TestLegacyImport:
    def test_old_file_sample_moves_into_the_database(self, tmp_path):
        """Перенос старой выборки: это часы работы оператора с рулеткой, терять нельзя."""
        post_dir = tmp_path / "posts" / "p1"
        (post_dir / "training").mkdir(parents=True)
        samples_path = post_dir / "training" / "samples.json"
        samples_path.write_text(
            json.dumps([json.loads(make_sample(i).model_dump_json()) for i in range(3)]),
            encoding="utf-8",
        )

        database = Database(tmp_path / "pallet.db")
        store = TrainingStore(database, "p1", post_dir)

        assert len(store.list()) == 3
        assert not samples_path.exists(), "старый файл должен быть переименован"
        assert samples_path.with_suffix(".json.imported").exists(), "но не удалён"

        # Повторное открытие не должно задваивать выборку.
        assert len(TrainingStore(database, "p1", post_dir).list()) == 3
        database.close()

    def test_broken_old_file_does_not_crash_the_post(self, tmp_path):
        post_dir = tmp_path / "posts" / "p1"
        (post_dir / "training").mkdir(parents=True)
        (post_dir / "training" / "samples.json").write_text("{не json", encoding="utf-8")

        database = Database(tmp_path / "pallet.db")
        assert TrainingStore(database, "p1", post_dir).list() == []
        database.close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import app as app_module
    from src.store import PostStore
    from src.worker import PostWorkerRegistry

    app_module.registry.stop_all()
    app_module.store = PostStore(tmp_path / "posts.json")
    app_module.database = Database(tmp_path / "pallet.db")
    app_module.POSTS_DIR = tmp_path / "posts"
    app_module.registry = PostWorkerRegistry(app_module.database, app_module.POSTS_DIR)
    yield TestClient(app_module.app), app_module
    app_module.registry.stop_all()
    app_module.database.close()


def create_post(client) -> str:
    res = client.post("/api/posts", json={
        "name": "Пост обучения", "camera_top_device_id": 91, "camera_side_device_id": 92,
    })
    assert res.status_code == 200, res.text
    return res.json()["id"]


class TestApi:
    def test_status_starts_untrained(self, client):
        api, _ = client
        post_id = create_post(api)
        body = api.get(f"/api/posts/{post_id}/training").json()
        assert body["samples_count"] == 0
        assert body["trained"] is False
        assert body["minimum_samples"] == MIN_SAMPLES

    def test_unknown_post_is_404(self, client):
        api, _ = client
        assert api.get("/api/posts/нет-такого/training").status_code == 404

    def test_cannot_mark_a_stopped_post(self, client):
        """Размечать нечего, пока пост не запущен и не видит объект."""
        api, _ = client
        post_id = create_post(api)
        res = api.post(f"/api/posts/{post_id}/training",
                       json={"height_mm": 100, "width_mm": 200, "depth_mm": 300})
        assert res.status_code == 409

    def test_training_needs_enough_samples(self, client):
        api, app_module = client
        post_id = create_post(api)
        store = TrainingStore(app_module.database, post_id)
        for i in range(MIN_SAMPLES - 1):
            store.add(make_sample(i))

        res = api.post(f"/api/posts/{post_id}/training/fit")
        assert res.status_code == 409
        assert str(MIN_SAMPLES) in res.json()["detail"]

    def test_full_cycle_train_and_clear(self, client):
        api, app_module = client
        post_id = create_post(api)
        store = TrainingStore(app_module.database, post_id)
        for i in range(40):
            store.add(make_sample(i))

        trained = api.post(f"/api/posts/{post_id}/training/fit").json()
        assert trained["trained"] is True
        assert trained["expected_error_mm"] is not None
        assert api.get(f"/api/posts/{post_id}/training").json()["samples_count"] == 40

        cleared = api.delete(f"/api/posts/{post_id}/training").json()
        assert cleared["samples_count"] == 0
        assert cleared["trained"] is False

    def test_measurements_journal_endpoint(self, client):
        api, app_module = client
        post_id = create_post(api)
        app_module.database.add_measurement(post_id, BACKEND_ORTHO, 100.0, 200.0, 300.0,
                                            raw=(1.0, 2.0, 3.0), features=[0.5])

        body = api.get(f"/api/posts/{post_id}/measurements").json()
        assert body["total"] == 1
        assert body["items"][0]["width_mm"] == 200.0

    def test_measurements_of_unknown_post_are_404(self, client):
        api, _ = client
        assert api.get("/api/posts/нет-такого/measurements").status_code == 404


class TestWorkerIntegration:
    def make_worker(self, db, tmp_path, **post_kwargs):
        from src.models import CameraSideBinding, CameraTopBinding, Post
        from src.worker import PostWorker

        post = Post(id="p", name="p", camera_top=CameraTopBinding(device_id=0),
                    camera_side=CameraSideBinding(device_id=1),
                    created_at=datetime.now(timezone.utc), **post_kwargs)
        return PostWorker(post, db, tmp_path)

    def test_captures_features_of_the_current_measurement(self, db, tmp_path):
        from src.vision.visual_hull import HullBox

        worker = self.make_worker(db, tmp_path)
        with pytest.raises(LookupError):
            worker.capture_training_sample(100.0, 200.0, 300.0)

        worker._last_sample = ([1.0] * 9, HullBox(105.0, 210.0, 315.0))
        sample = worker.capture_training_sample(100.0, 200.0, 300.0)
        assert sample.height_mm == 100.0
        assert sample.measured_height_mm == 105.0
        assert sample.backend == BACKEND_ORTHO

    def test_correction_is_loaded_and_reloaded(self, db, tmp_path):
        worker = self.make_worker(db, tmp_path)
        worker.reload_correction()
        assert worker._correction is None

        for i in range(MIN_SAMPLES):
            worker.training.add(make_sample(i))
        worker.training.train(BACKEND_ORTHO)
        worker.reload_correction()
        assert worker._correction is not None

    def test_disabled_correction_is_not_loaded(self, db, tmp_path):
        worker = self.make_worker(db, tmp_path, correction_enabled=False)
        for i in range(MIN_SAMPLES):
            worker.training.add(make_sample(i))
        worker.training.train(BACKEND_ORTHO)
        worker.reload_correction()
        assert worker._correction is None


class TestLegacyModelImport:
    """Модель переносится отдельно от выборки — иначе поправка молча перестаёт применяться."""

    def write_model(self, post_dir, store_with_model):
        model = store_with_model.load_model()
        (post_dir / "training").mkdir(parents=True, exist_ok=True)
        (post_dir / "training" / "correction.json").write_text(
            json.dumps(model.to_dict()), encoding="utf-8")

    def test_model_moves_even_when_samples_are_already_in_the_database(self, tmp_path):
        database = Database(tmp_path / "pallet.db")
        trained = TrainingStore(database, "p1")
        for i in range(MIN_SAMPLES):
            trained.add(make_sample(i))
        trained.train(BACKEND_ORTHO)

        post_dir = tmp_path / "posts" / "p2"
        self.write_model(post_dir, trained)
        # У p2 выборки в файле нет вовсе — только модель, как после прерванного переноса.
        store = TrainingStore(database, "p2", post_dir)

        assert store.load_model() is not None
        assert store.status(BACKEND_ORTHO).trained is True
        assert (post_dir / "training" / "correction.json.imported").exists()
        database.close()
