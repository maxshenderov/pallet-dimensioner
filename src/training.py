"""Размеченная выборка и обученная поправка одной точки измерения.

Оператор ставит паллету, вводит истинные габариты рулеткой, сервис забирает признаки текущего
измерения и складывает пару «признаки → истина». После ~40 таких пар модель обучается и дальше
правит габариты автоматически.

Хранение — в общей базе (`storage.Database`), а не в файлах рядом с точкой: выборка это часы
работы оператора с рулеткой, и переписывание файла целиком на каждую запись оставляло окно, в
котором обрыв питания обнуляет всё. Старые файлы `training/samples.json` переносятся в базу
автоматически при первом обращении.

Признаки сохраняются и при КАЖДОМ замере в журнал (`Database.add_measurement`), а не только по
кнопке «Запомнить»: тогда модель можно переобучить задним числом, не выставляя паллеты заново.
Истинные габариты часто уже известны из заказа в 1С.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from .models import TrainingSample, TrainingStatus
from .storage import Database
from .vision.correction import (
    MIN_SAMPLES,
    RECOMMENDED_SAMPLES,
    CorrectionModel,
    cross_validated_error,
    fit,
    sample_warning,
)

logger = logging.getLogger(__name__)


class TrainingStore:
    """Выборка и модель одной точки поверх общей базы."""

    def __init__(self, database: Database, post_id: str, post_dir: str | Path | None = None):
        self._db = database
        self._post_id = post_id
        if post_dir is not None:
            self._db.import_legacy_training(post_id, post_dir)

    # -- замеры --------------------------------------------------------------

    def list(self, backend: str | None = None) -> list[TrainingSample]:
        return [
            TrainingSample(
                features=row["features"],
                height_mm=row["height_mm"], width_mm=row["width_mm"], depth_mm=row["depth_mm"],
                backend=row["backend"],
                measured_height_mm=row["measured_height_mm"],
                measured_width_mm=row["measured_width_mm"],
                measured_depth_mm=row["measured_depth_mm"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in self._db.training_samples(self._post_id, backend)
        ]

    def add(self, sample: TrainingSample) -> int:
        self._db.add_training_sample(
            self._post_id, sample.backend, list(sample.features),
            sample.height_mm, sample.width_mm, sample.depth_mm,
            measured=(sample.measured_height_mm, sample.measured_width_mm, sample.measured_depth_mm),
            created_at=sample.created_at,
        )
        return len(self._db.training_samples(self._post_id, sample.backend))

    def clear(self) -> None:
        self._db.clear_training(self._post_id)

    # -- модель --------------------------------------------------------------

    def load_model(self) -> CorrectionModel | None:
        row = self._db.latest_model(self._post_id)
        if row is None:
            return None
        try:
            return CorrectionModel.from_dict(row["model"])
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Не читается модель поправки точки %s: %s", self._post_id, e)
            return None

    def train(self, backend: str, guarantee_no_underestimate: bool = False) -> TrainingStatus:
        """Обучает поправку по накопленным замерам и сохраняет её в базу.

        Возвращает ожидаемую ошибку на НЕвиданных паллетах (по блокам), а не на обучающих
        данных: вторая всегда выглядит лучше и скрывает переобучение.
        """
        samples = self.list(backend)
        if len(samples) < MIN_SAMPLES:
            raise ValueError(
                f"Для обучения нужно минимум {MIN_SAMPLES} замеров этим методом, есть {len(samples)}"
            )

        features = np.array([s.features for s in samples], dtype=np.float64)
        truths = np.array([[s.height_mm, s.width_mm, s.depth_mm] for s in samples], dtype=np.float64)

        model = fit(features, truths, backend=backend,
                    guarantee_no_underestimate=guarantee_no_underestimate)
        # Сила регуляризации берётся из обученной модели, а не по умолчанию: иначе оценка
        # считалась бы для ДРУГОЙ модели, чем сохранённая, и показывала бы не ту ошибку.
        expected = cross_validated_error(features, truths, backend=backend,
                                         ridge_lambda=model.ridge_lambda,
                                         guarantee_no_underestimate=guarantee_no_underestimate)

        self._db.add_model(self._post_id, backend, len(samples), model.to_dict(), expected)
        if expected is None:
            logger.info("Поправка обучена на %d замерах; на оценку качества замеров пока не хватает",
                        len(samples))
        else:
            logger.info("Поправка обучена на %d замерах, ожидаемая ошибка В/Ш/Г %.0f/%.0f/%.0f мм",
                        len(samples), *expected)
        return self.status(backend)

    def status(self, backend: str) -> TrainingStatus:
        """Оценка качества берётся из базы, а не пересчитывается.

        Раньше она жила только в ответе на обучение и пропадала после перезапуска: модель
        работает, а насколько ей верить — уже не видно.
        """
        samples = self.list(backend)
        row = self._db.latest_model(self._post_id, backend)

        warning = None
        if len(samples) >= 2:
            warning = sample_warning(
                np.array([s.features for s in samples], dtype=np.float64),
                np.array([[s.height_mm, s.width_mm, s.depth_mm] for s in samples], dtype=np.float64),
                backend,
            )
        return TrainingStatus(
            backend=backend,
            samples_count=len(samples),
            minimum_samples=MIN_SAMPLES,
            recommended_samples=RECOMMENDED_SAMPLES,
            trained=row is not None,
            trained_on_samples=row["samples_count"] if row else None,
            expected_error_mm=row["expected_error_mm"] if row else None,
            warning=warning,
            updated_at=datetime.fromisoformat(row["trained_at"]) if row else None,
        )
