"""Тесты integration/api_client.py: успешная отправка, retry, буферизация при сбое, flush буфера."""
from datetime import datetime, timezone

import httpx
import respx

from src.integration.api_client import flush_pending_events, send_measurement
from src.models import MeasurementEvent

ENDPOINT = "http://api.test/pallet-measurements"


def _event() -> MeasurementEvent:
    return MeasurementEvent(
        post_id="POST-TEST",
        timestamp=datetime(2026, 8, 18, 10, 23, 41, tzinfo=timezone.utc),
        length_mm=1195,
        width_mm=800,
        height_mm=1487,
        cross_check_passed=True,
        cross_check_delta_mm=8,
        samples_count=12,
    )


@respx.mock
def test_send_measurement_success_no_buffering(tmp_path):
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200))

    ok = send_measurement(_event(), ENDPOINT, pending_events_dir=tmp_path)

    assert ok is True
    assert route.called
    assert list(tmp_path.glob("*.json")) == []


@respx.mock
def test_send_measurement_buffers_after_retries_exhausted(tmp_path):
    respx.post(ENDPOINT).mock(return_value=httpx.Response(500))

    ok = send_measurement(
        _event(), ENDPOINT, retry_attempts=2, retry_backoff_seconds=0.01, pending_events_dir=tmp_path
    )

    assert ok is False
    buffered = list(tmp_path.glob("*.json"))
    assert len(buffered) == 1


@respx.mock
def test_flush_pending_events_resends_and_clears_buffer(tmp_path):
    event = _event()
    (tmp_path / "pending.json").write_text(event.model_dump_json(), encoding="utf-8")
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200))

    flush_pending_events(ENDPOINT, tmp_path)

    assert route.called
    assert list(tmp_path.glob("*.json")) == []


@respx.mock
def test_flush_pending_events_keeps_buffer_on_repeated_failure(tmp_path):
    event = _event()
    (tmp_path / "pending.json").write_text(event.model_dump_json(), encoding="utf-8")
    respx.post(ENDPOINT).mock(return_value=httpx.Response(500))

    flush_pending_events(ENDPOINT, tmp_path)

    assert list(tmp_path.glob("*.json")) == [tmp_path / "pending.json"]
