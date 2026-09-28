"""Обучаемая поправка к габаритам: ставим паллету, вводим истинные размеры, снимаем, учим.

Работает над ЛЮБЫМ методом измерения — различаются только наборы признаков:

* `ortho_features` — нынешний метод передней грани (две камеры, top + side). Съедает его
  систематику: утопленный в корпус оптический центр, наклон верхней камеры, приближение
  «надир = центр кадра». Ту самую, которую иначе приходится вылавливать рулеткой.
* `hull_features` — вокселная резка (см. visual_hull.py). Резка даёт ВНЕШНЮЮ оценку: занизить
  габарит она не может, но систематически завышает, и завышает по-разному в зависимости от
  формы груза. На синтетике из 210 паллет сырая резка даёт 54/66/94 мм (высота/ширина/глубина),
  после поправки — 14/30/28.

Сорока размеченных паллет уже хватает, дальше кривая выполаживается.

Модель — гребневая регрессия на девяти признаках и их квадратах, матрица примерно 20×3. Ни
TensorFlow, ни sklearn (их в venv нет и не будет: TF не собран под Python 3.14), только numpy.
Нейросети на этих данных проверялись и проигрывают: 34/49/40 против 14/30/28. Причина не в
слабости сети, а в объёме данных — 40-150 примеров и 9 числовых признаков это задача для
линейной модели. Простая линейная поправка по одному габариту резки даёт уже 16/38/37,
то есть почти весь выигрыш; потолок задают признаки, а не сложность модели.

ПЛАТА ЗА ТОЧНОСТЬ. Сырая резка никогда не занижает — для WMS это ценное свойство. Обученная
поправка его теряет: занижает примерно в 55 % случаев (худшее наблюдавшееся — 159 мм по ширине).
Вернуть гарантию можно запасом (safety_margin_mm), но точность откатывается почти к исходной.
Либо точность втрое, либо гарантия не занижать — одновременно не выйдет, это решение заказчика.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .visual_hull import HullBox

logger = logging.getLogger(__name__)

# Ниже этого числа образцов обучение отказывает. Порог не запас прочности, а измеренный факт:
# на десяти замерах поправка делает ХУЖЕ, чем её отсутствие (средняя ошибка 142 мм против 50),
# потому что параметров у модели больше, чем данных.
MIN_SAMPLES = 20
# Начиная отсюда кривая точности выполаживается — столько паллет и стоит размечать руками.
RECOMMENDED_SAMPLES = 40
LAYER_COUNT = 5
FEATURE_COUNT = 4 + LAYER_COUNT

# Поправка не знает, каким методом получен габарит: обучение одинаково работает и над вокселной
# резкой, и над нынешним методом передней грани. Различаются только наборы признаков, поэтому
# модель помнит, для какого бэкенда обучена, и отказывается применяться к чужим признакам.
BACKEND_HULL = "visual_hull"
BACKEND_ORTHO = "dual_webcam_ortho"

# Признак, не менявшийся по всей выборке, ничему научить не может — зато способен всё сломать.
# При стандартизации он делится на почти нулевой разброс, и на паллете, хоть немного отличной от
# обучающих, множитель вырастает на порядки: выборка «одна коробка на одном месте» так давала
# габариты в сотни метров при истинных сотнях миллиметров. Такие столбцы глушатся огромным
# масштабом — их вклад становится нулевым и при обучении, и при применении.
DEAD_COLUMN_SCALE = 1e30
MIN_RELATIVE_SPREAD = 1e-6

# Минимальный размер обучающей доли при перекрёстной проверке и минимальное число блоков,
# по которым оценка вообще имеет смысл.
SELECTION_MIN_TRAIN = 10
MIN_FOLDS = 3


def hull_features(grid: np.ndarray, occupancy: np.ndarray, box: HullBox) -> np.ndarray:
    """Признаки одного измерения: габарит резки, заполненность и профиль площади по ярусам.

    Профиль — то, что отличает ролик на паллете от сплошной коробки: у ролика нижний ярус
    широкий, а верхние узкие. Без него модель не знает, какого рода груз перед ней, и может
    только сдвинуть все габариты на одну общую величину.
    """
    points = grid[occupancy]
    if len(points) == 0:
        return np.zeros(FEATURE_COUNT)

    voxel_mm = _voxel_size(grid)
    footprint = max(box.width_mm * box.depth_mm, 1.0)
    heights = points[:, 2]

    profile = []
    for index in range(LAYER_COUNT):
        low, high = box.height_mm * index / LAYER_COUNT, box.height_mm * (index + 1) / LAYER_COUNT
        in_layer = (heights >= low) & (heights < high + 1e-9)
        profile.append(float(in_layer.sum()) * voxel_mm**2 / footprint)

    fill = len(points) * voxel_mm**3 / max(footprint * box.height_mm, 1.0)
    return np.array([box.height_mm, box.width_mm, box.depth_mm, fill, *profile])


def _voxel_size(grid: np.ndarray) -> float:
    """Шаг сетки восстанавливается из неё самой — отдельным параметром его таскать незачем."""
    unique = np.unique(grid[:, 2])
    return float(unique[1] - unique[0]) if len(unique) > 1 else 1.0


ORTHO_FEATURE_NAMES = (
    "height_mm", "width_mm", "depth_mm", "side_mm", "distance_near_mm",
    "offset_from_nadir_mm", "footprint_fill", "aspect", "raw_long_side_mm",
)


def ortho_features(
    height_mm: float,
    width_mm: float,
    depth_mm: float,
    side_mm: float,
    distance_near_mm: float,
    floor_points_mm: np.ndarray,
    nadir_mm: np.ndarray,
) -> np.ndarray:
    """Признаки одного измерения методом передней грани (нынешний бэкенд поста).

    Помимо самого габарита сюда входит всё, от чего зависит остаточная ошибка этого метода:

    * `distance_near_mm` — расстояние до передней грани. Рулетка до объектива даёт
      систематический промах (оптический центр утоплен в корпус, на стенде — 36 мм), и он
      входит в результат пропорционально расстоянию.
    * `offset_from_nadir_mm` — насколько груз стоит в стороне от точки под верхней камерой.
      Поправка на боковые грани приближает надир центром кадра и тем сильнее промахивается,
      чем дальше груз от центра; наклон верхней камеры добавляется туда же.
    * `footprint_fill`, `aspect`, `raw_long_side_mm` — форма следа на полу и величина самой
      перспективной поправки, то есть то, к чему эта поправка применялась.

    Регрессия по этим признакам и должна съесть систематику, которую иначе приходится
    вылавливать вручную (см. [[измерение_по_расстоянию]]).
    """
    points = np.asarray(floor_points_mm, dtype=np.float32).reshape(-1, 2)
    centre, (long_side, short_side), _ = cv2.minAreaRect(points)

    offset = float(np.hypot(centre[0] - float(nadir_mm[0]), centre[1] - float(nadir_mm[1])))
    footprint_area = float(cv2.contourArea(points))
    fill = footprint_area / max(width_mm * depth_mm, 1.0)

    return np.array([
        height_mm, width_mm, depth_mm, side_mm, distance_near_mm,
        offset, fill, width_mm / max(depth_mm, 1.0), max(long_side, short_side),
    ])


def sample_warning(features: np.ndarray, truths: np.ndarray, backend: str) -> str | None:
    """Чему выборка научить НЕ сможет — предупреждение оператору до и после обучения.

    Поправка верна только в тех пределах, которые ей показали. Обучение на одной паллете даёт
    модель, точную для этой паллеты и врущую на любой другой, причём молча: `cross_validated_error`
    такую выборку не ловит, потому что отложенные замеры в ней ровно так же однообразны, как
    обучающие, и ошибка на них выходит прекрасной.

    Проверки безразмерные — стенд в масштабе 1:8 и настоящий пост меряются одной меркой.
    """
    features = np.asarray(features, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    if len(truths) < 2:
        return None

    notes = []
    span = (truths.max(axis=0) - truths.min(axis=0)) / np.maximum(truths.mean(axis=0), 1.0)
    if span.max() < 0.15:
        notes.append("все замеры на паллетах одного размера — на других поправка будет врать")

    if backend == BACKEND_ORTHO:
        # Порог измерен, а не назначен: на замерах из одной точки размах удаления от надира
        # выходит около 0.10 от длинной стороны груза (и это не движение груза, а перспектива
        # верхней грани — она зависит от высоты), а на замерах по всей зоне — около 0.74.
        offset = features[:, ORTHO_FEATURE_NAMES.index("offset_from_nadir_mm")]
        long_side = features[:, ORTHO_FEATURE_NAMES.index("raw_long_side_mm")]
        if (offset.max() - offset.min()) < 0.3 * max(long_side.mean(), 1.0):
            notes.append("груз всё время стоял на одном месте — переставляй его по зоне")

    return "; ".join(notes) if notes else None


@dataclass(frozen=True)
class CorrectionModel:
    """Обученная поправка. Хранится в data/posts/{id}/correction.json."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    samples_count: int
    backend: str = BACKEND_HULL
    safety_margin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    clamp_to_raw: bool = True
    # По каким измерениям поправка вообще применяется. Она обучается по всем трём сразу, но
    # помогает не везде: на посту с 24 замерами по высоте и глубине она снимала ошибку втрое,
    # а по ширине УДВАИВАЛА её (18 мм против 9 у сырого замера). Измерение, где поправка не
    # лучше сырого, остаётся сырым — иначе обучение ухудшает результат и молчит об этом.
    apply_to: tuple[bool, bool, bool] = (True, True, True)
    ridge_lambda: float = 1e-3

    @property
    def feature_count(self) -> int:
        return len(self.mean) // 2

    def apply(self, features: np.ndarray, raw: HullBox) -> HullBox:
        """Исправленный габарит по сырому измерению raw.

        Для вокселной резки поправка может только СЖИМАТЬ (clamp_to_raw): резка гарантированно
        не занижает, поэтому её результат — безопасный потолок, и на незнакомой форме груза
        регрессия не сможет выдать произвольно большое число.

        Для метода передней грани такого потолка нет: он ошибается в обе стороны, и ограничивать
        поправку сверху нечем — там clamp_to_raw выключается.
        """
        features = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if features.shape[1] != self.feature_count:
            raise ValueError(f"Модель обучена на {self.feature_count} признаках, получено {features.shape[1]}")

        raw_values = np.array([raw.height_mm, raw.width_mm, raw.depth_mm])
        predicted = (_design(features, self.mean, self.scale) @ self.weights).ravel()
        corrected = predicted + np.asarray(self.safety_margin_mm)
        if self.clamp_to_raw:
            corrected = np.minimum(corrected, raw_values)
        corrected = np.where(self.apply_to, corrected, raw_values)
        return HullBox(float(corrected[0]), float(corrected[1]), float(corrected[2]))

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "samples_count": self.samples_count,
            "backend": self.backend,
            "safety_margin_mm": list(self.safety_margin_mm),
            "clamp_to_raw": self.clamp_to_raw,
            "apply_to": list(self.apply_to),
            "ridge_lambda": self.ridge_lambda,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CorrectionModel:
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            weights=np.asarray(data["weights"], dtype=np.float64),
            samples_count=int(data["samples_count"]),
            backend=data.get("backend", BACKEND_HULL),
            safety_margin_mm=tuple(data.get("safety_margin_mm", (0.0, 0.0, 0.0))),
            clamp_to_raw=bool(data.get("clamp_to_raw", True)),
            apply_to=tuple(bool(v) for v in data.get("apply_to", (True, True, True))),
            ridge_lambda=float(data.get("ridge_lambda", 1e-3)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> CorrectionModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _design(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Признаки и их квадраты, стандартизованные, плюс свободный член.

    Свободный член добавляется ПОСЛЕ стандартизации: у столбца единиц нулевая дисперсия, и при
    делении на неё он обнуляется вместе со всей константной частью модели. Симптом — предсказания
    около нуля при внешне корректном обучении.
    """
    extended = np.hstack([features, features**2])
    return np.hstack([(extended - mean) / scale, np.ones((len(features), 1))])


def cross_validated_error(features: np.ndarray, truths: np.ndarray, folds: int = 5,
                          min_train_size: int = SELECTION_MIN_TRAIN,
                          **fit_kwargs) -> tuple[float, float, float] | None:
    """Ожидаемая ошибка поправки на НЕвиданных паллетах, мм по высоте/ширине/глубине.

    Считается по k блокам: модель учится на четырёх пятых замеров и проверяется на оставшейся.
    Ошибка на самих обучающих данных всегда выглядит хорошо и обманывает — оператору надо
    показывать именно эту оценку, иначе переобучение проявится уже на настоящих паллетах.

    Порог на размер обучающей доли намеренно НЕ равен MIN_SAMPLES. Сначала он был им, и на
    24 замерах из пяти блоков проходил ровно один: оценка считалась по четырём отложенным
    замерам и прыгала как угодно — по ширине выходило 15 мм там, где полная пятиблочная оценка
    даёт 8.9. Одна доля хуже, чем доля поменьше, но усреднённая по пяти.

    Возвращает None, когда не набирается даже MIN_FOLDS блоков: тогда честнее не показывать
    ничего, чем показывать число, посчитанное по горстке замеров.
    """
    features = np.asarray(features, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)

    errors = []
    for fold in range(folds):
        held_out = np.arange(len(features)) % folds == fold
        if held_out.all() or (~held_out).sum() < min_train_size:
            continue
        model = _fit_core(features[~held_out], truths[~held_out], **fit_kwargs)
        predicted = np.array([
            list(model.apply(row, HullBox(*row[:3]))) for row in features[held_out]
        ])
        errors.append(np.abs(predicted - truths[held_out]))

    if len(errors) < MIN_FOLDS:
        return None
    mean_error = np.vstack(errors).mean(axis=0)
    return float(mean_error[0]), float(mean_error[1]), float(mean_error[2])


# Сетка силы регуляризации. Слабая регуляризация даёт большие веса, а вместе с квадратичными
# признаками — колоссальное усиление шума: на живом посту 0.2 мм дрожания сырого замера
# превращались в 9 мм дрожания исправленного. Точность при этом почти не выигрывает.
RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0)
# Дрожание сырого замера, замеренное на работающем посту: габарит гуляет на десятые доли мм.
NOISE_PROBE_MM = 0.2
# Сколько дрожания допустимо на выходе. Цифра, скачущая на экране, для оператора хуже, чем
# цифра, смещённая на столько же: по прыгающему габариту он не понимает, когда замер закончен,
# и стабилизатор не сходится. Отсюда бюджет заметно строже точности модели.
NOISE_BUDGET_MM = 1.0


def _noise_gain(model: CorrectionModel, features: np.ndarray) -> float:
    """Насколько поправка раздувает шум измерения, мм на выходе при NOISE_PROBE_MM на входе."""
    rng = np.random.default_rng(0)
    base = np.asarray(features, dtype=np.float64).mean(axis=0)
    # Дрожат первые три признака — сам габарит, он и в миллиметрах. Остальные признаки у разных
    # методов измерения в разных единицах (доля заполнения, отношение сторон), и трясти их на
    # «0.2 мм» бессмысленно: для безразмерной величины около единицы это огромное возмущение.
    probes = np.tile(base, (120, 1))
    probes[:, :3] += rng.normal(0, NOISE_PROBE_MM, size=(120, 3))
    outputs = [list(model.apply(sample, HullBox(*sample[:3]))) for sample in probes]
    return float(np.max(np.std(outputs, axis=0)))


def choose_ridge_lambda(features: np.ndarray, truths: np.ndarray, **fit_kwargs) -> float:
    """Подбор силы регуляризации по отложенным замерам И по дрожанию на выходе.

    Одной точности мало. Слабая регуляризация вместе с квадратичными признаками даёт огромные
    веса: на живом посту 0.2 мм дрожания сырого замера превращались в 9 мм дрожания
    исправленного, замер переставал стабилизироваться, и оператор не мог понять, когда цифру
    можно записывать. Поэтому сначала отбираются варианты, укладывающиеся в бюджет дрожания,
    и уже среди них берётся самый точный.
    """
    scored = []
    for candidate in RIDGE_GRID:
        error = cross_validated_error(features, truths, ridge_lambda=candidate, **fit_kwargs)
        if error is None:
            continue
        model = _fit_core(features, truths, ridge_lambda=candidate, **fit_kwargs)
        scored.append((candidate, float(np.mean(error)), _noise_gain(model, features)))
    if not scored:
        return RIDGE_GRID[0]

    steady = [row for row in scored if row[2] <= NOISE_BUDGET_MM]
    if steady:
        return min(steady, key=lambda row: row[1])[0]
    return min(scored, key=lambda row: row[2])[0]  # ни один не тихий — берём самый тихий


def _useful_dimensions(features: np.ndarray, truths: np.ndarray,
                       ridge_lambda: float, **fit_kwargs) -> tuple[bool, bool, bool]:
    """По каким измерениям поправка действительно лучше сырого замера.

    Обучение идёт по всем трём сразу, но помогает не везде: на посту с 24 замерами по высоте и
    глубине поправка снимала ошибку втрое, а по ширине УДВАИВАЛА её. Молча ухудшать измерение
    нельзя, поэтому такое измерение остаётся сырым.
    """
    corrected = cross_validated_error(features, truths, ridge_lambda=ridge_lambda, **fit_kwargs)
    if corrected is None:
        return (True, True, True)
    raw = np.abs(features[:, :3] - truths).mean(axis=0)
    return tuple(bool(c < r) for c, r in zip(corrected, raw))


def fit(features: np.ndarray, truths: np.ndarray, ridge_lambda: float | None = None,
        guarantee_no_underestimate: bool = False, backend: str = BACKEND_HULL,
        clamp_to_raw: bool | None = None) -> CorrectionModel:
    """Обучение поправки по размеченным паллетам.

    features — (N, K) из hull_features или ortho_features, truths — (N, 3) истинные
    высота/ширина/глубина. guarantee_no_underestimate добавляет запас, при котором модель не
    занижает ни на одном образце обучения; точность при этом падает примерно вдвое.

    `ridge_lambda=None` — подобрать по отложенным замерам, это и есть штатный режим. Явное
    значение нужно самому подбору и тестам, иначе получилась бы рекурсия.
    """
    features = np.asarray(features, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    if features.ndim == 2 and len(features) == len(truths):
        features, truths, dropped = drop_unusable(features, truths)
        if dropped:
            logger.warning("Из обучения исключено %d нефизичных замеров", dropped)
    _validate(features, truths)
    core_kwargs = dict(guarantee_no_underestimate=guarantee_no_underestimate,
                       backend=backend, clamp_to_raw=clamp_to_raw)

    if ridge_lambda is None:
        ridge_lambda = choose_ridge_lambda(features, truths, **core_kwargs)

    model = _fit_core(features, truths, ridge_lambda=ridge_lambda, **core_kwargs)
    return replace(model, apply_to=_useful_dimensions(features, truths, ridge_lambda, **core_kwargs))


def drop_unusable(features: np.ndarray, truths: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Выбрасывает нефизичные замеры: это не данные, а сбой измерения.

    Один такой замер портит всю модель. Пример из проверки: груз оказался почти вплотную к
    объективу, глубина вышла −3.3 мм, а признак заполнения следа — 3.3 миллиона вместо единицы.
    Матрица после этого вырождается, и ошибка модели уходит на десять порядков (10^12 мм),
    причём одинаково при любой силе регуляризации.
    """
    features = np.asarray(features, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    keep = (
        np.isfinite(features).all(axis=1)
        & np.isfinite(truths).all(axis=1)
        & (features[:, :3] > 0).all(axis=1)
        & (truths > 0).all(axis=1)
    )
    return features[keep], truths[keep], int((~keep).sum())


def _validate(features: np.ndarray, truths: np.ndarray) -> None:
    """Проверки уровня всего обучения.

    Порог MIN_SAMPLES проверяется только здесь, а не в `_fit_core`: доли перекрёстной проверки
    заведомо меньше выборки, и требовать от каждой полного порога значило бы запретить подбор.
    """
    if len(features) < MIN_SAMPLES:
        raise ValueError(f"Для обучения нужно минимум {MIN_SAMPLES} размеченных паллет, есть {len(features)}")
    if features.ndim != 2 or features.shape[1] < 3:
        raise ValueError("Признаки должны быть матрицей (N, K), где первые три — сырой габарит")
    if len(features) != len(truths):
        raise ValueError(f"Признаков {len(features)}, а истинных габаритов {len(truths)}")


def _fit_core(features: np.ndarray, truths: np.ndarray, ridge_lambda: float = 1e-3,
              guarantee_no_underestimate: bool = False, backend: str = BACKEND_HULL,
              clamp_to_raw: bool | None = None) -> CorrectionModel:
    """Собственно гребневая регрессия при заданной силе регуляризации."""
    features = np.asarray(features, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    if clamp_to_raw is None:
        clamp_to_raw = backend == BACKEND_HULL

    extended = np.hstack([features, features**2])
    mean, spread = extended.mean(axis=0), extended.std(axis=0)
    informative = spread > MIN_RELATIVE_SPREAD * np.maximum(np.abs(mean), 1.0)
    if not informative.any():
        raise ValueError(
            "Все замеры одинаковые — учиться не на чем. Переставляй паллету по зоне, "
            "поворачивай её и бери разные размеры."
        )
    scale = np.where(informative, spread, DEAD_COLUMN_SCALE)

    design = _design(features, mean, scale)
    penalty = np.eye(design.shape[1])
    penalty[-1, -1] = 0.0  # свободный член не штрафуем, иначе модель тянет предсказания к нулю
    weights = np.linalg.solve(design.T @ design + ridge_lambda * len(design) * penalty, design.T @ truths)

    if not guarantee_no_underestimate:
        return CorrectionModel(mean, scale, weights, len(features), backend,
                               clamp_to_raw=clamp_to_raw, ridge_lambda=ridge_lambda)

    # Запас считается по НЕобрезанному предсказанию — ровно так, как его потом применит apply():
    # там сначала прибавляется запас и только затем срабатывает ограничение сверху по резке.
    # Если считать запас по уже обрезанному предсказанию, apply() обрежет сумму обратно к резке,
    # и гарантия «не занижать» не выполнится.
    margin = np.maximum((truths - design @ weights).max(axis=0), 0.0)
    return CorrectionModel(
        mean, scale, weights, len(features), backend,
        tuple(float(v) for v in margin), clamp_to_raw, ridge_lambda=ridge_lambda,
    )
