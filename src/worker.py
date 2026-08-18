"""Фоновый воркер точки измерения: захват камер, пайплайн измерения, live-состояние для веб-слоя."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import httpx

from .calibration.aruco_homography import compute_homography, detect_markers
from .calibration.camera_calibrate import load_intrinsics, undistort
from .capture.camera_stream import CameraStream, CameraUnavailableError, get_synchronized_frames
from .integration.api_client import send_measurement
from .models import MeasurementEvent, Post, PostState, PostStatus, WeightReading
from .vision.dimensions import compute_footprint_mm, compute_height_and_side_mm, correct_perspective, cross_check
from .vision.segmentation import segment_pallet
from .vision.stabilizer import MeasurementStabilizer

logger = logging.getLogger(__name__)

CAMERA_OFFLINE_RETRY_SECONDS = 5.0
CALIBRATION_LOST_RETRY_SECONDS = 1.0
WEIGHT_TIMEOUT_SECONDS = 2.0
JPEG_QUALITY = 80


class PostWorker:
    """Один воркер = один физический пост (2 камеры + опционально весы), в своём потоке."""

    def __init__(self, post: Post, intrinsics_dir: str | Path = "data/posts"):
        self.post = post
        self._intrinsics_dir = Path(intrinsics_dir) / post.id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = PostState(post_id=post.id)
        self._frames: dict[str, bytes] = {}

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._set_state(status=PostStatus.starting)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"post-{self.post.id}")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._set_state(status=PostStatus.stopped)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- readers (called from web request threads) --------------------------

    def get_state(self) -> PostState:
        with self._lock:
            return self._state.model_copy()

    def get_frame_jpeg(self, camera: str) -> bytes | None:
        with self._lock:
            return self._frames.get(camera)

    # -- internals ------------------------------------------------------------

    def _set_state(self, **kwargs) -> None:
        with self._lock:
            self._state = self._state.model_copy(update={**kwargs, "updated_at": _now()})

    def _set_frame(self, camera: str, frame) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        with self._lock:
            self._frames[camera] = buf.tobytes()

    def _fetch_weight(self) -> WeightReading | None:
        if not self.post.scale_api_url:
            return None
        try:
            response = httpx.get(f"{self.post.scale_api_url.rstrip('/')}/api/weight", timeout=WEIGHT_TIMEOUT_SECONDS)
            response.raise_for_status()
            return WeightReading(**response.json())
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("Весы недоступны (%s): %s", self.post.scale_api_url, e)
            return WeightReading(ok=False)

    def _load_intrinsics(self, camera_id: str):
        try:
            return load_intrinsics(self._intrinsics_dir, camera_id)
        except (FileNotFoundError, OSError):
            return None

    def _run(self) -> None:
        post = self.post
        intrinsics_top = self._load_intrinsics("top")
        intrinsics_side = self._load_intrinsics("side")

        stream_top = None
        try:
            stream_top = CameraStream(post.camera_top.device_id, post.camera_top.resolution)
            stream_side = CameraStream(post.camera_side.device_id, post.camera_side.resolution)
        except CameraUnavailableError as e:
            if stream_top is not None:
                stream_top.release()
            logger.error("Пост %s: камера недоступна при старте: %s", post.id, e)
            self._set_state(status=PostStatus.camera_offline, error=str(e))
            return

        stabilizer = MeasurementStabilizer(post.stabilizer_window_size, post.stabilizer_stddev_threshold_mm)
        logger.info("Пост %s запущен", post.id)
        self._set_state(status=PostStatus.running, error=None)

        try:
            while not self._stop_event.is_set():
                self._tick(post, stream_top, stream_side, intrinsics_top, intrinsics_side, stabilizer)
        finally:
            stream_top.release()
            stream_side.release()
            logger.info("Пост %s остановлен", post.id)

    def _tick(self, post, stream_top, stream_side, intrinsics_top, intrinsics_side, stabilizer) -> None:
        try:
            frame_top, frame_side = get_synchronized_frames(stream_top, stream_side)
        except CameraUnavailableError as e:
            logger.error("Пост %s camera_offline: %s", post.id, e)
            self._set_state(status=PostStatus.camera_offline, error=str(e))
            time.sleep(CAMERA_OFFLINE_RETRY_SECONDS)
            return

        if intrinsics_top is not None:
            frame_top = undistort(frame_top, intrinsics_top)
        if intrinsics_side is not None:
            frame_side = undistort(frame_side, intrinsics_side)

        self._set_frame("top", frame_top)
        self._set_frame("side", frame_side)
        weight = self._fetch_weight()

        homography_top = compute_homography(detect_markers(frame_top), post.camera_top.aruco_marker_positions_mm)
        homography_side = compute_homography(detect_markers(frame_side), post.camera_side.aruco_marker_positions_mm)

        if homography_top is None or homography_side is None:
            stabilizer.reset()
            self._set_state(status=PostStatus.calibration_lost, weight=weight, stable=False)
            time.sleep(CALIBRATION_LOST_RETRY_SECONDS)
            return

        contour_top = segment_pallet(
            frame_top, post.camera_top.roi,
            post.camera_top.background_hsv_lower, post.camera_top.background_hsv_upper,
            post.camera_top.min_contour_area_px,
        )
        contour_side = segment_pallet(
            frame_side, post.camera_side.roi,
            post.camera_side.background_hsv_lower, post.camera_side.background_hsv_upper,
            post.camera_side.min_contour_area_px,
        )

        if contour_top is None or contour_side is None:
            stabilizer.reset()
            self._set_state(status=PostStatus.running, weight=weight, stable=False,
                             length_mm=None, width_mm=None, height_mm=None, cross_check_passed=None)
            return

        footprint = compute_footprint_mm(contour_top, homography_top)
        height, side = compute_height_and_side_mm(contour_side, homography_side, post.camera_side.floor_line_px)
        length, width = correct_perspective(
            footprint, height, post.camera_top.mount_height_mm, post.camera_top.mount_height_mm
        )
        passed = cross_check((length, width), side, post.cross_check_tolerance_mm)

        stabilizer.add_sample(length, width, height)
        is_stable = stabilizer.is_stable()

        self._set_state(
            status=PostStatus.running, weight=weight, stable=is_stable,
            length_mm=length, width_mm=width, height_mm=height, cross_check_passed=passed,
        )

        if is_stable:
            result = stabilizer.get_stable_result()
            if post.wms_endpoint:
                delta_mm = min(abs(length - side), abs(width - side))
                event = MeasurementEvent(
                    post_id=post.id, timestamp=_now(),
                    length_mm=result.length_mm, width_mm=result.width_mm, height_mm=result.height_mm,
                    cross_check_passed=passed, cross_check_delta_mm=delta_mm, samples_count=result.samples_count,
                )
                send_measurement(event, post.wms_endpoint, pending_events_dir=f"data/posts/{post.id}/pending_events")
                logger.info("Пост %s: измерение отправлено в WMS L=%.0f W=%.0f H=%.0f",
                            post.id, result.length_mm, result.width_mm, result.height_mm)
            stabilizer.reset()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PostWorkerRegistry:
    """Держит активные PostWorker по post_id — общий реестр для FastAPI-приложения."""

    def __init__(self):
        self._workers: dict[str, PostWorker] = {}
        self._lock = threading.Lock()

    def get_or_create(self, post: Post) -> PostWorker:
        with self._lock:
            worker = self._workers.get(post.id)
            if worker is None or worker.post != post:
                worker = PostWorker(post)
                self._workers[post.id] = worker
            return worker

    def get(self, post_id: str) -> PostWorker | None:
        with self._lock:
            return self._workers.get(post_id)

    def remove(self, post_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(post_id, None)
        if worker:
            worker.stop()

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.stop()
