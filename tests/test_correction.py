"""Тесты обучаемой поправки (vision/correction.py) на синтетических паллетах."""
from __future__ import annotations

import numpy as np
import pytest

from src.vision.correction import (
    cross_validated_error,
    BACKEND_ORTHO,
    FEATURE_COUNT,
    MIN_SAMPLES,
    ORTHO_FEATURE_NAMES,
    NOISE_BUDGET_MM,
    RIDGE_GRID,
    _fit_core,
    _noise_gain,
    CorrectionModel,
    fit,
    hull_features,
    ortho_features,
    sample_warning,
)
from src.vision.perspective import (
    footprint_mm_without_side_faces,
    height_mm_from_near_face,
    object_distances_mm,
    width_mm_from_pinhole,
)
from src.vision.synthetic import Box, Cylinder, look_at_pose, pinhole_matrix, render_silhouette
from src.vision.visual_hull import HullBox, bounding_box_mm, carve, project_grid, voxel_grid

IMAGE_SIZE = (640, 360)
FOV_DEG = 90.0
VOXEL_MM = 40.0
ZONE_MM = (2000.0, 1500.0)
VOLUME_MM = 2200.0
CAMERAS = [(-900.0, 0.0, 4000.0), (900.0, 0.0, 4000.0), (4500.0, 0.0, 1200.0), (0.0, 4500.0, 1200.0)]

GRID = voxel_grid(ZONE_MM, VOLUME_MM, VOXEL_MM)
CAMERA_MATRIX = pinhole_matrix(IMAGE_SIZE, FOV_DEG)
POSES = [look_at_pose(p, (0.0, 0.0, VOLUME_MM / 2)) for p in CAMERAS]
# Позы поста неподвижны, поэтому проекции сетки считаются один раз — так же, как это будет
# в воркере между пересчётами позы по красному прямоугольнику.
PROJECTIONS = [project_grid(GRID, rvec, tvec, CAMERA_MATRIX) for rvec, tvec in POSES]


def measure(scene) -> tuple[np.ndarray, HullBox] | None:
    masks = [render_silhouette(scene, IMAGE_SIZE, rvec, tvec, CAMERA_MATRIX) for rvec, tvec in POSES]
    occupancy = carve(PROJECTIONS, masks)
    box = bounding_box_mm(GRID, occupancy, VOXEL_MM)
    if box is None:
        return None
    return hull_features(GRID, occupancy, box), box


def random_pallet(rng, family):
    cx, cy = rng.uniform(-350, 350), rng.uniform(-250, 250)
    if family == "box":
        w, d, h = rng.uniform(700, 1300), rng.uniform(500, 900), rng.uniform(400, 1800)
        return [Box((cx, cy), (w, d), 0.0, h, rng.uniform(0, 90))], (h, max(w, d), min(w, d))
    if family == "two_level":
        w, d, h = rng.uniform(900, 1300), rng.uniform(600, 900), rng.uniform(800, 1800)
        base = rng.uniform(120, 350)
        return ([Box((cx, cy), (w, d), 0.0, base),
                 Box((cx, cy), (w * 0.6, d * 0.6), base, h)], (h, max(w, d), min(w, d)))
    w, d, h = rng.uniform(1000, 1300), rng.uniform(700, 900), rng.uniform(1200, 2000)
    return ([Box((cx, cy), (w, d), 0.0, 144.0),
             Cylinder((cx, cy), rng.uniform(220, 380), 144.0, h)], (h, max(w, d), min(w, d)))


def dataset(rng, count):
    families = ("box", "two_level", "roll")
    features, truths, hulls = [], [], []
    for index in range(count):
        scene, truth = random_pallet(rng, families[index % 3])
        result = measure(scene)
        if result is None:
            continue
        features.append(result[0])
        truths.append(truth)
        hulls.append(result[1])
    return np.array(features), np.array(truths), hulls


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    return dataset(rng, 90), dataset(rng, 30)


class TestFeatures:
    def test_shape_and_leading_dimensions(self):
        features, box = measure([Box((0.0, 0.0), (1200.0, 800.0), 0.0, 1500.0)])
        assert features.shape == (FEATURE_COUNT,)
        assert features[:3] == pytest.approx([box.height_mm, box.width_mm, box.depth_mm])

    def test_profile_tells_a_roll_from_a_box(self):
        """Профиль площади по ярусам — единственный признак, по которому модель понимает,
        что за груз перед ней. У ролика верхние ярусы узкие, у коробки — как нижние."""
        roll, _ = measure([Box((0.0, 0.0), (1200.0, 800.0), 0.0, 144.0),
                           Cylinder((0.0, 0.0), 300.0, 144.0, 2000.0)])
        box, _ = measure([Box((0.0, 0.0), (1200.0, 800.0), 0.0, 2000.0)])
        assert roll[-1] / roll[4] < box[-1] / box[4]

    def test_empty_occupancy_gives_zero_features(self):
        empty = np.zeros(len(GRID), dtype=bool)
        assert hull_features(GRID, empty, HullBox(0.0, 0.0, 0.0)).tolist() == [0.0] * FEATURE_COUNT


class TestFit:
    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match=str(MIN_SAMPLES)):
            fit(np.zeros((MIN_SAMPLES - 1, FEATURE_COUNT)), np.zeros((MIN_SAMPLES - 1, 3)))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="истинных"):
            fit(np.zeros((20, FEATURE_COUNT)), np.zeros((19, 3)))

    def test_model_rejects_foreign_features(self, data):
        """Модель помнит, на каком наборе признаков обучена: признаки резки и признаки метода
        передней грани не взаимозаменяемы, а по длине могут совпасть."""
        (features, truths, _), _ = data
        model = fit(features, truths)
        with pytest.raises(ValueError, match="обучена"):
            model.apply(np.zeros(FEATURE_COUNT + 2), HullBox(1.0, 1.0, 1.0))

    def test_learning_beats_raw_carving(self, data):
        """Главный тест: ради этого поправка и существует."""
        (features, truths, _), (test_features, test_truths, test_hulls) = data
        model = fit(features, truths)

        raw = np.abs(test_features[:, :3] - test_truths).mean(axis=0)
        corrected = np.abs(
            np.array([list(model.apply(f, h)) for f, h in zip(test_features, test_hulls)]) - test_truths
        ).mean(axis=0)

        assert (corrected < raw).all(), f"сырая {raw}, после поправки {corrected}"
        assert corrected.mean() < raw.mean() / 1.5

    def test_forty_samples_are_enough(self, data):
        """Кривая точности выполаживается около сорока паллет — столько и размечаем руками."""
        (features, truths, _), (test_features, test_truths, test_hulls) = data

        def error(count):
            model = fit(features[:count], truths[:count])
            predicted = np.array([list(model.apply(f, h)) for f, h in zip(test_features, test_hulls)])
            return np.abs(predicted - test_truths).mean()

        assert error(40) < error(MIN_SAMPLES)
        assert error(len(features)) < error(40) * 1.3


class TestApply:
    def test_never_exceeds_the_hull(self, data):
        """Поправка только сжимает: на незнакомой форме регрессия может выдать что угодно,
        а резка гарантированно не занижает и потому служит безопасным потолком."""
        (features, truths, _), _ = data
        model = fit(features, truths)
        absurd = np.zeros(FEATURE_COUNT)
        hull = HullBox(1000.0, 900.0, 800.0)
        corrected = model.apply(absurd, hull)
        assert corrected.height_mm <= hull.height_mm
        assert corrected.width_mm <= hull.width_mm
        assert corrected.depth_mm <= hull.depth_mm

    def test_safety_margin_stops_underestimating(self, data):
        """Гарантия «не занижать» возвращается запасом — ценой точности.

        Поправка ограничена сверху результатом резки, поэтому и гарантия наследуется от неё:
        занизить сильнее самой резки модель с запасом не может. Резка на дискретной сетке
        промахивается максимум на воксель, отсюда допуск.
        """
        (features, truths, hulls), _ = data
        safe = fit(features, truths, guarantee_no_underestimate=True)
        predicted = np.array([list(safe.apply(f, h)) for f, h in zip(features, hulls)])

        floor = np.minimum(truths, features[:, :3])
        assert (predicted >= floor - VOXEL_MM).all()
        assert any(margin > 0 for margin in safe.safety_margin_mm)

    def test_safety_margin_costs_accuracy(self, data):
        """Либо точность втрое, либо гарантия не занижать — одновременно не выйдет."""
        (features, truths, _), (test_features, test_truths, test_hulls) = data
        sharp = fit(features, truths)
        safe = fit(features, truths, guarantee_no_underestimate=True)

        def error(model):
            predicted = np.array([list(model.apply(f, h)) for f, h in zip(test_features, test_hulls)])
            return np.abs(predicted - test_truths).mean()

        assert error(safe) > error(sharp)


MOUNT_MM = 3200.0
FOCAL_PX = 1400.0
LENS_DEPTH_MM = -1200.0
# Верхняя камера смотрит не строго вниз, а пайплайн приближает надир центром кадра.
TRUE_NADIR = np.array([140.0, -90.0])
ASSUMED_NADIR = np.array([0.0, 0.0])
# Рулетка до объектива промахивается: оптический центр утоплен в корпус. На стенде — 36 мм.
DISTANCE_BIAS_MM = 36.0


def ortho_sample(rng, width=None, depth=None, height=None, cx=None, cy=None):
    """Одно измерение нынешним методом — со всеми его систематическими промахами.

    Размер и место можно задать явно: так собирается однообразная выборка (одна и та же паллета,
    всегда на одном месте), на которой поправка обязана отказаться обучаться.

    Воспроизводит стенд: объектив боковой камеры отмерен рулеткой с ошибкой, верхняя камера
    слегка наклонена, надир приближён центром кадра. Габарит от этого «плывёт» тем сильнее,
    чем дальше груз от центра и чем он выше.
    """
    width = rng.uniform(400, 1400) if width is None else width
    depth = rng.uniform(300, 900) if depth is None else depth
    height = rng.uniform(200, 1600) if height is None else height
    cx = rng.uniform(-600, 600) if cx is None else cx
    cy = rng.uniform(-400, 400) if cy is None else cy

    magnification = MOUNT_MM / (MOUNT_MM - height)
    base = np.array([[cx - width / 2, cy - depth / 2], [cx + width / 2, cy - depth / 2],
                     [cx + width / 2, cy + depth / 2], [cx - width / 2, cy + depth / 2]])
    top = (base - TRUE_NADIR) * magnification + TRUE_NADIR
    floor_points = np.vstack([base, top]).astype(np.float32)

    near_true, _ = object_distances_mm(floor_points, LENS_DEPTH_MM, "y")
    near_assumed, _ = object_distances_mm(floor_points, LENS_DEPTH_MM + DISTANCE_BIAS_MM, "y")

    # Боковая камера снимает переднюю грань на истинном расстоянии, а пайплайн пересчитывает
    # её по ошибочному — отсюда систематический промах, пропорциональный расстоянию.
    height_px = height * FOCAL_PX / near_true
    side_px = width * FOCAL_PX / near_true
    height_measured = height_mm_from_near_face(0.0, height_px, near_assumed, FOCAL_PX)
    side_measured = width_mm_from_pinhole(side_px, near_assumed, FOCAL_PX)
    width_measured, depth_measured = footprint_mm_without_side_faces(
        floor_points, ASSUMED_NADIR, height_measured, MOUNT_MM
    )

    features = ortho_features(height_measured, width_measured, depth_measured,
                              side_measured, near_assumed, floor_points, ASSUMED_NADIR)
    measured = HullBox(height_measured, width_measured, depth_measured)
    return features, measured, (height, max(width, depth), min(width, depth))


def ortho_dataset(rng, count):
    rows = [ortho_sample(rng) for _ in range(count)]
    return (np.array([r[0] for r in rows]), np.array([r[2] for r in rows]), [r[1] for r in rows])


@pytest.fixture(scope="module")
def ortho_data():
    rng = np.random.default_rng(17)
    return ortho_dataset(rng, 90), ortho_dataset(rng, 40)


class TestOrthoBackend:
    """Обучение поверх метода передней грани — того, что уже работает на посту."""

    def test_features_have_expected_names(self):
        features, _, _ = ortho_sample(np.random.default_rng(1))
        assert features.shape == (len(ORTHO_FEATURE_NAMES),)

    def test_learning_removes_the_systematic_error(self, ortho_data):
        (features, truths, _), (test_features, test_truths, test_measured) = ortho_data
        model = fit(features, truths, backend=BACKEND_ORTHO)

        raw = np.abs(test_features[:, :3] - test_truths).mean(axis=0)
        corrected = np.abs(
            np.array([list(model.apply(f, m)) for f, m in zip(test_features, test_measured)]) - test_truths
        ).mean(axis=0)

        assert (corrected < raw).all(), f"сырое {raw}, после поправки {corrected}"

    def test_correction_is_not_clamped_for_this_backend(self, ortho_data):
        """Метод передней грани ошибается в обе стороны, ограничивать поправку сверху нечем —
        в отличие от резки, которая заведомо завышает."""
        (features, truths, _), _ = ortho_data
        assert fit(features, truths, backend=BACKEND_ORTHO).clamp_to_raw is False
        assert fit(features, truths).clamp_to_raw is True

    def test_helps_most_where_the_load_stands_off_centre(self, ortho_data):
        """Систематика метода растёт с удалением груза от точки под камерой — именно её
        обучение и должно съесть."""
        (features, truths, _), (test_features, test_truths, test_measured) = ortho_data
        model = fit(features, truths, backend=BACKEND_ORTHO)
        far = test_features[:, 5] > np.median(test_features[:, 5])

        raw = np.abs(test_features[far, :3] - test_truths[far]).mean()
        corrected = np.abs(
            np.array([list(model.apply(f, m)) for f, m, keep
                      in zip(test_features, test_measured, far) if keep]) - test_truths[far]
        ).mean()
        assert corrected < raw / 2


class TestCrossValidation:
    def test_estimates_error_on_unseen_pallets(self, ortho_data):
        """Оценка по блокам должна быть близка к ошибке на честно отложенной выборке."""
        (features, truths, _), (test_features, test_truths, test_measured) = ortho_data
        estimated = cross_validated_error(features, truths, backend=BACKEND_ORTHO)

        model = fit(features, truths, backend=BACKEND_ORTHO)
        actual = np.abs(
            np.array([list(model.apply(f, m)) for f, m in zip(test_features, test_measured)]) - test_truths
        ).mean(axis=0)

        assert np.allclose(estimated, actual, atol=25.0), f"оценка {estimated}, факт {actual}"

    def test_is_worse_than_error_on_training_data(self, ortho_data):
        """Ошибка на обучающих данных всегда льстит — показывать оператору надо не её."""
        (features, truths, measured), _ = ortho_data
        model = fit(features, truths, backend=BACKEND_ORTHO)
        on_training = np.abs(
            np.array([list(model.apply(f, m)) for f, m in zip(features, measured)]) - truths
        ).mean()
        assert np.mean(cross_validated_error(features, truths, backend=BACKEND_ORTHO)) > on_training


class TestPersistence:
    def test_round_trip_through_json(self, tmp_path, data):
        (features, truths, hulls), _ = data
        model = fit(features, truths)
        loaded = CorrectionModel.load(model.save(tmp_path / "correction.json"))

        assert loaded.samples_count == model.samples_count
        assert list(loaded.apply(features[0], hulls[0])) == pytest.approx(list(model.apply(features[0], hulls[0])))


class TestSampleDiversity:
    """Однообразная выборка — самая опасная: оценка по отложенным замерам её не ловит.

    Отложенные замеры в такой выборке ровно так же однообразны, ошибка на них выходит
    прекрасной, и негодная поправка выглядит обученной.
    """

    def one_pallet_one_spot(self, count, jitter=0.0):
        rng = np.random.default_rng(3)
        rows = [
            ortho_sample(rng, width=800.0, depth=600.0, height=900.0,
                         cx=rng.uniform(-jitter, jitter), cy=rng.uniform(-jitter, jitter))
            for _ in range(count)
        ]
        return (np.array([r[0] for r in rows]), np.array([r[2] for r in rows]),
                [r[1] for r in rows])

    def test_identical_samples_are_refused(self):
        features, truths, _ = self.one_pallet_one_spot(MIN_SAMPLES)
        with pytest.raises(ValueError, match="одинаковые"):
            fit(features, truths, backend=BACKEND_ORTHO)

    def test_nearly_identical_samples_do_not_explode(self):
        """Ровно та поломка, ради которой глушатся неизменные столбцы.

        Признак с почти нулевым разбросом делится на этот разброс, и на паллете, чуть отличной
        от обучающих, множитель вырастает на порядки: получались габариты в сотни метров.
        """
        features, truths, _ = self.one_pallet_one_spot(MIN_SAMPLES, jitter=0.5)
        model = fit(features, truths, backend=BACKEND_ORTHO)

        other, measured, _ = ortho_sample(np.random.default_rng(11), width=1200.0, depth=800.0,
                                          height=1400.0, cx=500.0, cy=300.0)
        predicted = np.array(list(model.apply(other, measured)))
        assert np.all(predicted < 10_000.0), f"поправка выдала {predicted}"

    def test_warns_when_all_pallets_are_the_same_size(self):
        features, truths, _ = self.one_pallet_one_spot(MIN_SAMPLES, jitter=400.0)
        assert "одного размера" in sample_warning(features, truths, BACKEND_ORTHO)

    def test_warns_when_the_load_never_moves(self):
        rng = np.random.default_rng(5)
        rows = [ortho_sample(rng, cx=0.0, cy=0.0) for _ in range(MIN_SAMPLES)]
        features = np.array([r[0] for r in rows])
        truths = np.array([r[2] for r in rows])
        assert "на одном месте" in sample_warning(features, truths, BACKEND_ORTHO)

    def test_varied_sample_is_not_flagged(self, ortho_data):
        (features, truths, _), _ = ortho_data
        assert sample_warning(features, truths, BACKEND_ORTHO) is None


class TestStabilityAndUsefulness:
    """Найдено на живом посту: сырой замер дрожал на 0.2 мм, а исправленный — на 9."""

    def ortho_set(self, count=40):
        rng = np.random.default_rng(23)
        rows = [ortho_sample(rng) for _ in range(count)]
        return np.array([r[0] for r in rows]), np.array([r[2] for r in rows])

    def test_weak_regularisation_amplifies_noise(self):
        """Проверка самой болезни: без подбора шум измерения раздувается на порядок."""
        features, truths = self.ortho_set()
        # Берётся сама регрессия, без отключения бесполезных измерений: иначе на выборке, где
        # поправка не помогает вовсе, обе модели вырождаются в сырой габарит и сравнивать нечего.
        weak = _fit_core(features, truths, ridge_lambda=1e-3, backend=BACKEND_ORTHO)
        strong = _fit_core(features, truths, ridge_lambda=10.0, backend=BACKEND_ORTHO)

        assert _noise_gain(strong, features) < _noise_gain(weak, features)

    def test_chosen_lambda_keeps_the_output_steady(self):
        """Проверяется свойство, а не выбранное число: важно, что цифра на экране не скачет.

        На какой именно силе регуляризации это достигается — дело подбора; на разных выборках
        подходит разная, и на этой годится уже самая слабая.
        """
        features, truths = self.ortho_set()
        chosen = fit(features, truths, backend=BACKEND_ORTHO)
        assert chosen.ridge_lambda in RIDGE_GRID
        assert _noise_gain(chosen, features) < NOISE_BUDGET_MM

    def test_correction_is_switched_off_where_it_does_not_help(self):
        """Измерение, которое поправка портит, обязано остаться сырым.

        Здесь глубина измеряется уже точно, и учить по ней нечему — модель на такой мишени
        только добавляет разброс.
        """
        features, truths = self.ortho_set()
        truths = truths.copy()
        truths[:, 2] = features[:, 2]  # глубина и так идеальна

        model = fit(features, truths, backend=BACKEND_ORTHO)
        assert model.apply_to[2] is False

        raw = HullBox(*features[0][:3])
        assert model.apply(features[0], raw).depth_mm == pytest.approx(raw.depth_mm)

    def test_switched_off_dimension_survives_a_round_trip(self, tmp_path):
        features, truths = self.ortho_set()
        truths = truths.copy()
        truths[:, 2] = features[:, 2]
        model = fit(features, truths, backend=BACKEND_ORTHO)

        loaded = CorrectionModel.load(model.save(tmp_path / "correction.json"))
        assert loaded.apply_to == model.apply_to
        assert loaded.ridge_lambda == model.ridge_lambda
