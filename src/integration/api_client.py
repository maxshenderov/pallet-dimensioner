"""Отправка событий измерения в REST API: retry с экспоненциальной задержкой + буферизация на диск."""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import httpx

from ..models import MeasurementEvent

logger = logging.getLogger(__name__)


def send_measurement(
    event: MeasurementEvent,
    endpoint_url: str,
    retry_attempts: int = 5,
    retry_backoff_seconds: float = 2.0,
    pending_events_dir: str | Path = "data/pending_events",
) -> bool:
    """
    POST события на endpoint_url с retry (экспоненциальная задержка).

    При успехе также пытается дослать ранее буферизованные события. При исчерпании
    retry — буферизует событие на диск для повторной отправки при следующем успешном соединении.
    """
    if _post(event, endpoint_url):
        flush_pending_events(endpoint_url, pending_events_dir)
        return True

    for attempt in range(1, retry_attempts):
        time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
        if _post(event, endpoint_url):
            flush_pending_events(endpoint_url, pending_events_dir)
            return True

    _buffer_event(event, pending_events_dir)
    return False


def _post(event: MeasurementEvent, endpoint_url: str) -> bool:
    try:
        response = httpx.post(endpoint_url, json=json.loads(event.model_dump_json()), timeout=10.0)
        response.raise_for_status()
        return True
    except httpx.HTTPError as e:
        logger.warning("Не удалось отправить измерение: %s", e)
        return False


def _buffer_event(event: MeasurementEvent, pending_events_dir: str | Path) -> None:
    directory = Path(pending_events_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{event.timestamp.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    path.write_text(event.model_dump_json(indent=2), encoding="utf-8")
    logger.error("Не удалось отправить измерение, событие буферизовано: %s", path)


def flush_pending_events(endpoint_url: str, pending_events_dir: str | Path) -> None:
    """Повторно отправляет буферизованные события; успешно отправленные — удаляет с диска."""
    directory = Path(pending_events_dir)
    if not directory.exists():
        return

    for path in sorted(directory.glob("*.json")):
        try:
            event = MeasurementEvent(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Битый файл в буфере событий, пропущен: %s (%s)", path, e)
            continue

        if _post(event, endpoint_url):
            path.unlink()
        else:
            break
