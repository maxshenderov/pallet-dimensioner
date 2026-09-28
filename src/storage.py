"""Постоянное хранилище поста: журнал замеров, размеченная выборка, обученные модели.

ПОЧЕМУ SQLITE, А НЕ JSON. Настройки точек (`posts.json`) — файл на десяток килобайт, который
переписывается раз в месяц, там JSON уместен. Журнал замеров устроен иначе: он растёт всё время
работы поста, и переписывание файла целиком на каждую запись даёт квадратичную стоимость и окно,
в котором обрыв питания оставляет обрезанный файл. Именно так и терялась бы выборка обучения —
несколько часов работы оператора с рулеткой. SQLite пишет через журнал упреждающей записи: либо
транзакция целиком, либо ничего.

ПОЧЕМУ НЕ ВНЕШНЯЯ БД. Пост — это коробка с камерами в цехе. При недоступной сети она обязана
продолжать мерить; внешняя БД добавила бы отказ, который останавливает измерение. Отправка
наружу — отдельная задача (`integration/api_client.py` с очередью повторов).

ГДЕ ФАЙЛ. В каталоге данных, который в Docker примонтирован томом (`./data:/app/data`) — база
переживает пересборку образа и пересоздание контейнера. Путь задаётся переменной `DATA_DIR`.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id            TEXT    NOT NULL,
    measured_at        TEXT    NOT NULL,
    backend            TEXT    NOT NULL,
    height_mm          REAL    NOT NULL,
    width_mm           REAL    NOT NULL,
    depth_mm           REAL    NOT NULL,
    raw_height_mm      REAL    NOT NULL,
    raw_width_mm       REAL    NOT NULL,
    raw_depth_mm       REAL    NOT NULL,
    weight_kg          REAL,
    cross_check_passed INTEGER,
    correction_applied INTEGER NOT NULL DEFAULT 0,
    samples_count      INTEGER,
    features           TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS measurements_by_post ON measurements(post_id, id DESC);

CREATE TABLE IF NOT EXISTS training_samples (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id            TEXT    NOT NULL,
    created_at         TEXT    NOT NULL,
    backend            TEXT    NOT NULL,
    height_mm          REAL    NOT NULL,
    width_mm           REAL    NOT NULL,
    depth_mm           REAL    NOT NULL,
    measured_height_mm REAL,
    measured_width_mm  REAL,
    measured_depth_mm  REAL,
    features           TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_by_post ON training_samples(post_id, backend);

CREATE TABLE IF NOT EXISTS correction_models (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id           TEXT    NOT NULL,
    trained_at        TEXT    NOT NULL,
    backend           TEXT    NOT NULL,
    samples_count     INTEGER NOT NULL,
    expected_error_mm TEXT,
    model             TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS models_by_post ON correction_models(post_id, id DESC);
"""


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    """Одна база на весь сервис, одно соединение под замком.

    Соединение общее и `check_same_thread=False`, потому что писать в него будут одновременно
    поток воркера (журнал замеров) и потоки веб-слоя (разметка, обучение). Замок здесь дешевле
    пула: запись — событие раз в несколько секунд, а не горячий путь.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(self._path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        # WAL: запись идёт в отдельный журнал и переносится в базу целиком. Обрыв питания
        # посреди записи откатывается к последнему целому состоянию, а не рвёт файл.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._connection.execute(sql, params)
            self._connection.commit()
            return cursor

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._connection.execute(sql, params).fetchall()

    # -- журнал замеров ------------------------------------------------------

    def add_measurement(self, post_id: str, backend: str, height_mm: float, width_mm: float,
                        depth_mm: float, raw: tuple[float, float, float], features: list[float],
                        weight_kg: float | None = None, cross_check_passed: bool | None = None,
                        correction_applied: bool = False, samples_count: int | None = None) -> int:
        """Один стабилизировавшийся замер. Признаки сохраняются вместе с габаритом.

        Именно признаки делают журнал полезным задним числом: истинные размеры паллеты часто
        известны из заказа в 1С, и тогда выборку для обучения можно набрать из уже сделанных
        замеров, не выставляя паллеты руками заново.
        """
        cursor = self._execute(
            "INSERT INTO measurements (post_id, measured_at, backend, height_mm, width_mm, depth_mm,"
            " raw_height_mm, raw_width_mm, raw_depth_mm, weight_kg, cross_check_passed,"
            " correction_applied, samples_count, features)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (post_id, now_utc().isoformat(), backend, height_mm, width_mm, depth_mm,
             raw[0], raw[1], raw[2], weight_kg,
             None if cross_check_passed is None else int(cross_check_passed),
             int(correction_applied), samples_count, json.dumps(features)),
        )
        return int(cursor.lastrowid)

    def measurements(self, post_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        rows = self._query(
            "SELECT * FROM measurements WHERE post_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (post_id, limit, offset),
        )
        return [self._measurement_to_dict(row) for row in rows]

    def measurements_count(self, post_id: str) -> int:
        return int(self._query("SELECT COUNT(*) AS n FROM measurements WHERE post_id = ?",
                               (post_id,))[0]["n"])

    @staticmethod
    def _measurement_to_dict(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["features"] = json.loads(data["features"])
        data["correction_applied"] = bool(data["correction_applied"])
        if data["cross_check_passed"] is not None:
            data["cross_check_passed"] = bool(data["cross_check_passed"])
        return data

    # -- размеченная выборка -------------------------------------------------

    def add_training_sample(self, post_id: str, backend: str, features: list[float],
                            height_mm: float, width_mm: float, depth_mm: float,
                            measured: tuple[float, float, float] | None = None,
                            created_at: datetime | None = None) -> int:
        cursor = self._execute(
            "INSERT INTO training_samples (post_id, created_at, backend, height_mm, width_mm,"
            " depth_mm, measured_height_mm, measured_width_mm, measured_depth_mm, features)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (post_id, (created_at or now_utc()).isoformat(), backend, height_mm, width_mm, depth_mm,
             *(measured or (None, None, None)), json.dumps(features)),
        )
        return int(cursor.lastrowid)

    def training_samples(self, post_id: str, backend: str | None = None) -> list[dict]:
        sql = "SELECT * FROM training_samples WHERE post_id = ?"
        params: tuple = (post_id,)
        if backend is not None:
            sql += " AND backend = ?"
            params += (backend,)
        rows = self._query(sql + " ORDER BY id", params)
        return [dict(row, features=json.loads(row["features"])) for row in rows]

    def clear_training(self, post_id: str) -> None:
        self._execute("DELETE FROM training_samples WHERE post_id = ?", (post_id,))
        self._execute("DELETE FROM correction_models WHERE post_id = ?", (post_id,))

    # -- обученные модели ----------------------------------------------------

    def add_model(self, post_id: str, backend: str, samples_count: int, model: dict,
                  expected_error_mm: tuple[float, float, float] | None) -> int:
        """Модели не перезаписываются, а накапливаются.

        Старая модель — единственное, с чем можно сравнить новую, когда после переобучения
        габариты вдруг поехали. Строка занимает килобайты, обучение происходит раз в месяцы.
        """
        cursor = self._execute(
            "INSERT INTO correction_models (post_id, trained_at, backend, samples_count,"
            " expected_error_mm, model) VALUES (?,?,?,?,?,?)",
            (post_id, now_utc().isoformat(), backend, samples_count,
             json.dumps(list(expected_error_mm)) if expected_error_mm else None,
             json.dumps(model)),
        )
        return int(cursor.lastrowid)

    def latest_model(self, post_id: str, backend: str | None = None) -> dict | None:
        sql = "SELECT * FROM correction_models WHERE post_id = ?"
        params: tuple = (post_id,)
        if backend is not None:
            sql += " AND backend = ?"
            params += (backend,)
        rows = self._query(sql + " ORDER BY id DESC LIMIT 1", params)
        if not rows:
            return None
        row = dict(rows[0])
        row["model"] = json.loads(row["model"])
        row["expected_error_mm"] = json.loads(row["expected_error_mm"]) if row["expected_error_mm"] else None
        return row

    # -- перенос со старого файлового хранилища ------------------------------

    def import_legacy_training(self, post_id: str, post_dir: str | Path) -> int:
        """Переносит выборку и обученную модель из `data/posts/{id}/training/`, если они там остались.

        Файлы после переноса не удаляются, а переименовываются: это единственная копия ручных
        замеров с рулеткой, и стирать её из-за возможной ошибки переноса нельзя.

        Выборка и модель переносятся НЕЗАВИСИМО: перенос мог прерваться между ними, и тогда на
        следующем запуске выборка уже в базе, а поправка ещё в файле — при общей проверке она бы
        так и осталась непере несённой, и габариты молча вернулись бы к сырым.
        """
        self._import_legacy_model(post_id, post_dir)

        path = Path(post_dir) / "training" / "samples.json"
        if not path.exists() or self.training_samples(post_id):
            return 0
        try:
            items = json.loads(path.read_text(encoding="utf-8") or "[]")
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Не переносится старая выборка %s: %s", path, e)
            return 0

        for item in items:
            self.add_training_sample(
                post_id, item.get("backend", "dual_webcam_ortho"), item["features"],
                item["height_mm"], item["width_mm"], item["depth_mm"],
                measured=(item.get("measured_height_mm"), item.get("measured_width_mm"),
                          item.get("measured_depth_mm")),
                created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
            )
        path.rename(path.with_suffix(".json.imported"))
        logger.info("Пост %s: перенесено %d замеров из %s в базу", post_id, len(items), path.name)
        return len(items)

    def _import_legacy_model(self, post_id: str, post_dir: str | Path) -> bool:
        """Переносит и обученную модель.

        Без этого поправка после перехода на базу молча перестаёт применяться: выборка на месте,
        а модели нет, и габариты возвращаются к сырым — заметить это можно только по цифрам.
        """
        path = Path(post_dir) / "training" / "correction.json"
        if not path.exists() or self.latest_model(post_id):
            return False
        try:
            model = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Не переносится старая модель поправки %s: %s", path, e)
            return False

        self.add_model(post_id, model.get("backend", "dual_webcam_ortho"),
                       int(model.get("samples_count", 0)), model, None)
        path.rename(path.with_suffix(".json.imported"))
        logger.info("Пост %s: перенесена обученная поправка из %s", post_id, path.name)
        return True
