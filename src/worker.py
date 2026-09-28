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
from .models import MeasurementEvent, Post, PostState, PostStatus, TrainingSample, WeightReading
from .storage import Database
from .training import TrainingStore
from .vision.correction import BACKEND_ORTHO, CorrectionModel, ortho_features
from .vision.dimensions import cross_check
from .vision.visual_hull import HullBox
from .vision.perspective import (
    camera_nadir_mm,
    clip_below_floor,
    contour_to_floor_mm,
    floor_row_px,
    focal_length_px_from_markers,
    footprint_mm_without_side_faces,
    height_looks_inflated,
    height_mm_from_near_face,
    lens_depth_mm,
    object_distances_mm,
    width_mm_from_pinhole,
)
from .vision.segmentation import segment_pallet
from .vision.stabilizer import MeasurementStabilizer

logger = logging.getLogger(__name__)

CAMERA_OFFLINE_RETRY_SECONDS = 5.0
CALIBRATION_LOST_RETRY_SECONDS = 1.0
# На столько силуэт может не дотягиваться до основания, пока это ещё считается шумом маски.
BASE_NOT_REACHED_PX = 15
WEIGHT_TIMEOUT_SECONDS = 2.0
JPEG_QUALITY = 80


class PostWorker:
    """Один воркер = один физический пост (2 камеры + опционально весы), в своём потоке."""

    def __init__(self, post: Post, database: Database, intrinsics_dir: str | Path = "data/posts"):
        self.post = post
        self._db = database
        self._post_dir = Path(intrinsics_dir) / post.id
        self._intrinsics_dir = self._post_dir
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = PostState(post_id=post.id)
        self._frames: dict[str, bytes] = {}
        self._ema: dict[str, float] = {}
        self.training = TrainingStore(database, post.id, self._post_dir)
        # Признаки последнего измерения — их забирает кнопка «Запомнить» на рабочем месте,
        # когда оператор вводит истинные габариты с рулетки.
        self._last_sample: tuple[list[float], HullBox] | None = None
        self._correction: CorrectionModel | None = None

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

    def capture_training_sample(self, height_mm: float, width_mm: float, depth_mm: float) -> TrainingSample:
        """Снимает признаки текущего измерения и связывает их с истинными габаритами.

        Габарит берётся СЫРОЙ, до обученной поправки: иначе следующая модель будет учиться
        поверх предыдущей, и ошибки начнут накапливаться от переобучения к переобучению.
        """
        with self._lock:
            sample = self._last_sample
        if sample is None:
            raise LookupError("Нет текущего измерения: объект не найден или пост не запущен")

        features, raw = sample
        return TrainingSample(
            features=features, height_mm=height_mm, width_mm=width_mm, depth_mm=depth_mm,
            backend=BACKEND_ORTHO,
            measured_height_mm=raw.height_mm, measured_width_mm=raw.width_mm, measured_depth_mm=raw.depth_mm,
            created_at=_now(),
        )

    def reload_correction(self) -> None:
        """Перечитывает модель поправки — вызывается после обучения, без перезапуска поста."""
        self._correction = self.training.load_model() if self.post.correction_enabled else None

    # -- internals ------------------------------------------------------------

    def _set_state(self, **kwargs) -> None:
        with self._lock:
            self._state = self._state.model_copy(update={**kwargs, "updated_at": _now()})

    def _smooth(self, key: str, value: float) -> float:
        """Экспоненциальное сглаживание кадр-к-кадру. Используется и для отображения, и как
        вход стабилизатора — иначе шум сегментации (пиксельное дрожание контура) не даёт
        stddev-окну сойтись, и статус "stable" не наступает, даже когда объект неподвижен."""
        prev = self._ema.get(key)
        smoothed = value if prev is None else 0.3 * value + 0.7 * prev
        self._ema[key] = smoothed
        return smoothed

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
        self.reload_correction()
        if self._correction is not None:
            logger.info("Пост %s: поправка обучена на %d замерах", post.id, self._correction.samples_count)
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

        weight = self._fetch_weight()

        markers_top = detect_markers(frame_top)
        markers_side = detect_markers(frame_side)

        homography_top = compute_homography(markers_top, post.camera_top.aruco_marker_positions_mm)
        # Боковой камере гомография не нужна: её масштаб зависит от расстояния до объекта и
        # пересчитывается на каждом кадре. От маркеров 5/6 нужно только фокусное расстояние.
        focal_side = focal_length_px_from_markers(
            markers_side,
            post.camera_side.aruco_marker_positions_mm,
            post.camera_side.reference_distance_mm,
        )

        if homography_top is None or focal_side is None:
            stabilizer.reset()
            self._ema.clear()
            self._set_frame("top", _draw_overlay(frame_top, None, ["Калибровка потеряна: не видны маркеры"]))
            self._set_frame("side", _draw_overlay(frame_side, None, ["Калибровка потеряна: не видны маркеры"]))
            self._set_state(status=PostStatus.calibration_lost, weight=weight, stable=False)
            time.sleep(CALIBRATION_LOST_RETRY_SECONDS)
            return

        contour_top = segment_pallet(
            frame_top, post.camera_top.roi,
            post.camera_top.background_hsv_lower, post.camera_top.background_hsv_upper,
            post.camera_top.min_contour_area_px,
            markers_top.values(),
        )
        contour_side = segment_pallet(
            frame_side, post.camera_side.roi,
            post.camera_side.background_hsv_lower, post.camera_side.background_hsv_upper,
            post.camera_side.min_contour_area_px,
            markers_side.values(),
        )

        if contour_top is None or contour_side is None:
            stabilizer.reset()
            self._ema.clear()
            with self._lock:
                self._last_sample = None  # нечего запоминать: размечать можно только живой замер
            self._set_frame("top", _draw_overlay(frame_top, contour_top, ["Объект не найден"]))
            self._set_frame("side", _draw_overlay(frame_side, contour_side, ["Объект не найден"], post.camera_side.floor_line_px))
            self._set_state(status=PostStatus.running, weight=weight, stable=False,
                             width_mm=None, depth_mm=None, height_mm=None, cross_check_passed=None,
                             raw_width_mm=None, raw_depth_mm=None, raw_height_mm=None)
            return

        # Верхняя камера даёт не только footprint, но и положение объекта в мм на полу —
        # отсюда расстояние до боковой камеры, а значит и её масштаб именно для этого объекта.
        side_cfg = post.camera_side
        floor_points = contour_to_floor_mm(contour_top, homography_top)
        near_mm, _far_mm = object_distances_mm(
            floor_points,
            lens_depth_mm(side_cfg.markers_depth_mm, side_cfg.reference_distance_mm, side_cfg.depth_sign),
            side_cfg.depth_axis,
        )
        # Ниже плоскости основания груза быть не может — всё, что маска захватила там (тень,
        # отражение в глянцевом полу, обрывок упаковки у ножки), из силуэта убирается.
        floor_row = floor_row_px(side_cfg.floor_row_a, side_cfg.floor_row_b, near_mm)
        contour_side = clip_below_floor(contour_side, floor_row)

        _, top_px, width_px, height_px = cv2.boundingRect(contour_side)
        height = height_mm_from_near_face(top_px, top_px + height_px, near_mm, focal_side)
        side = width_mm_from_pinhole(width_px, near_mm, focal_side)

        if floor_row is not None and floor_row - (top_px + height_px) > BASE_NOT_REACHED_PX:
            # Контур не дотянулся до основания: низ груза в маску не попал. Обрезка это не
            # исправляет намеренно — иначе занижение высоты стало бы невидимым.
            logger.warning("Пост %s: низ груза не выделился, силуэт кончается на %d px выше "
                           "основания — высота занижена", post.id, floor_row - (top_px + height_px))

        if height_looks_inflated(height, side_cfg.lens_height_mm):
            logger.warning(
                "Пост %s: объект (%.0f мм) ниже объектива боковой камеры (%.0f мм) — "
                "камере видна верхняя грань, высота завышена. Опустите камеру.",
                post.id, height, side_cfg.lens_height_mm,
            )

        nadir = camera_nadir_mm(homography_top, (frame_top.shape[1], frame_top.shape[0]))
        # Ширина — вдоль оси x пола, глубина — вдоль y; оси задаёт таблица меток верхней
        # камеры. `side_cfg.depth_axis` сюда не входит: он про масштаб боковой камеры
        # (object_distances_mm выше), а не про названия габаритов.
        width, depth = footprint_mm_without_side_faces(
            floor_points, nadir, height, post.camera_top.mount_height_mm,
        )
        passed = cross_check((width, depth), side, post.cross_check_tolerance_mm)

        raw_width_disp = self._smooth("width", width)
        raw_depth_disp = self._smooth("depth", depth)
        raw_height_disp = self._smooth("height", height)
        raw = HullBox(raw_height_disp, raw_width_disp, raw_depth_disp)

        # Признаки считаются на КАЖДОМ кадре, а не только в режиме обучения: тогда модель можно
        # переобучить задним числом, не выставляя паллеты заново.
        features = ortho_features(
            raw_height_disp, raw_width_disp, raw_depth_disp,
            side, near_mm, floor_points, nadir,
        )
        with self._lock:
            self._last_sample = (features.tolist(), raw)

        corrected = raw
        if self._correction is not None:
            try:
                corrected = self._correction.apply(features, raw)
            except ValueError as e:
                logger.warning("Пост %s: поправка не применена (%s)", post.id, e)
                self._correction = None

        height_disp, width_disp, depth_disp = corrected

        self._set_frame("top", _draw_overlay(
            frame_top, contour_top, [f"Ширина: {width_disp:.0f} мм", f"Глубина: {depth_disp:.0f} мм"],
        ))
        self._set_frame("side", _draw_overlay(
            frame_side, contour_side, [f"Высота: {height_disp:.0f} мм"],
            int(floor_row) if floor_row is not None else post.camera_side.floor_line_px,
        ))

        stabilizer.add_sample(width_disp, depth_disp, height_disp)
        is_stable = stabilizer.is_stable()

        self._set_state(
            status=PostStatus.running, weight=weight, stable=is_stable,
            width_mm=width_disp, depth_mm=depth_disp, height_mm=height_disp, cross_check_passed=passed,
            raw_width_mm=raw_width_disp, raw_depth_mm=raw_depth_disp, raw_height_mm=raw_height_disp,
            correction_applied=self._correction is not None,
        )

        if is_stable:
            result = stabilizer.get_stable_result()

            # Замер уходит в журнал независимо от того, настроен ли WMS: отправка наружу может
            # быть не настроена или недоступна, а измерение всё равно уже сделано, и восстановить
            # его потом неоткуда. Пишется стабилизировавшийся результат, один на паллету, а не
            # каждый кадр.
            try:
                self._db.add_measurement(
                    post.id, BACKEND_ORTHO,
                    height_mm=result.height_mm, width_mm=result.length_mm, depth_mm=result.width_mm,
                    raw=(raw_height_disp, raw_width_disp, raw_depth_disp),
                    features=features.tolist(),
                    weight_kg=weight.value if weight and weight.ok else None,
                    cross_check_passed=passed,
                    correction_applied=self._correction is not None,
                    samples_count=result.samples_count,
                )
            except Exception as e:  # журнал не должен останавливать измерение
                logger.error("Пост %s: замер не записан в журнал: %s", post.id, e)

            if post.wms_endpoint:
                delta_mm = min(abs(width - side), abs(depth - side))
                event = MeasurementEvent(
                    post_id=post.id, timestamp=_now(),
                    height_mm=result.height_mm, width_mm=result.length_mm, depth_mm=result.width_mm,
                    cross_check_passed=passed, cross_check_delta_mm=delta_mm, samples_count=result.samples_count,
                )
                # Каталог очереди берётся от каталога точки, а не собирается от текущего рабочего:
                # при запуске не из папки сервиса неотправленные события уходили бы мимо тома.
                send_measurement(event, post.wms_endpoint,
                                 pending_events_dir=str(self._post_dir / "pending_events"))
                logger.info("Пост %s: измерение отправлено в WMS Ш=%.0f Г=%.0f В=%.0f",
                            post.id, result.length_mm, result.width_mm, result.height_mm)
            stabilizer.reset()


def _draw_overlay(frame, contour, lines: list[str], floor_line_px: int | None = None):
    """Рисует на кадре рамку найденного контура, текст с габаритами и (для side) линию пола."""
    out = frame.copy()
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if floor_line_px is not None:
        cv2.line(out, (0, floor_line_px), (out.shape[1], floor_line_px), (0, 165, 255), 1)
    y0 = 36
    for line in lines:
        cv2.putText(out, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        y0 += 36
    return out


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PostWorkerRegistry:
    """Держит активные PostWorker по post_id — общий реестр для FastAPI-приложения."""

    def __init__(self, database: Database, posts_dir: str | Path = "data/posts"):
        self._workers: dict[str, PostWorker] = {}
        self._db = database
        self._posts_dir = Path(posts_dir)
        self._lock = threading.Lock()

    def get_or_create(self, post: Post) -> PostWorker:
        with self._lock:
            worker = self._workers.get(post.id)
            if worker is None or worker.post != post:
                worker = PostWorker(post, self._db, self._posts_dir)
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
