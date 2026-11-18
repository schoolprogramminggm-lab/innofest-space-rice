"""Тесты классификатора зон (train_zones.py, zones.py). Интернет и настоящие снимки не нужны."""
import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import train_zones
import zones

N = len(zones.FEATURE_NAMES)    # 13 признаков
NDVI_JULY = zones.FEATURE_NAMES.index("ndvi_07")
SI_JULY = zones.FEATURE_NAMES.index("si_07")

# Искусственные «центры» классов: у каждого класса свой NDVI и SI в июле, остальное шум
CENTERS = {
    "healthy": (0.80, 0.03),
    "stress": (0.45, 0.05),
    "salinity": (0.10, 0.20),
    "not_rice": (0.05, 0.12),
}


def fake_features(cls, n, rng):
    """n пикселей класса cls: шум вокруг его центра."""
    X = rng.normal(0.1, 0.02, (n, N)).astype("float32")
    X[:, NDVI_JULY] = rng.normal(CENTERS[cls][0], 0.03, n)
    X[:, SI_JULY] = rng.normal(CENTERS[cls][1], 0.01, n)
    X[:, -1] = 1.0 if cls != "not_rice" else 0.0       # признак затопления
    return X


def fake_dataset(per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([fake_features(c, per_class, rng) for c in CENTERS])
    y = np.array([c for c in CENTERS for _ in range(per_class)])
    return X, y


# ---------- Обучение и предсказание ----------
def test_features_are_13_numbers():
    assert N == 13 and zones.FEATURE_NAMES[-1] == "flooded_spring"
    assert zones.FEATURE_NAMES[0] == "ndvi_05" and "lswi_08" in zones.FEATURE_NAMES


def test_train_and_predict_on_artificial_features():
    X, y = fake_dataset()
    model, result = train_zones.train_and_test(X, y)

    # 75/25 со стратификацией: во всех классах 10 из 40 точек в проверке
    assert sum(result["test"].values()) == 40 and sum(result["train"].values()) == 120
    assert all(result["test"][c] == 10 for c in CENTERS)
    assert result["accuracy"] > 0.95                      # классы на искусственных данных разделимы
    assert set(result["classes"]) == set(CENTERS)
    assert result["matrix"].shape == (4, 4) and result["matrix"].sum() == 40
    assert all(0 <= result["f1"][c] <= 1 for c in CENTERS)

    # предсказание: пиксель «как здоровый рис» -> healthy
    rng = np.random.default_rng(5)
    assert list(model.predict(fake_features("healthy", 3, rng))) == ["healthy"] * 3
    assert list(model.predict(fake_features("salinity", 3, rng))) == ["salinity"] * 3


def test_class_with_one_point_cannot_be_split_and_is_reported():
    X, y = fake_dataset()
    X = np.vstack([X, fake_features("healthy", 1, np.random.default_rng(1))])
    y = np.append(y, "water")                              # одна точка воды
    model, result = train_zones.train_and_test(X, y)
    assert result["too_few"] == ["water"] and "water" not in model.classes_
    assert result["total"]["water"] == 1


def test_training_needs_two_classes():
    X, y = fake_dataset()
    keep = y == "healthy"
    with pytest.raises(ValueError):
        train_zones.train_and_test(X[keep], y[keep])


# ---------- Сохранение и загрузка модели ----------
def test_save_and_load_model(tmp_path):
    X, y = fake_dataset()
    model, _ = train_zones.train_and_test(X, y)
    path = zones.save_model(model, tmp_path / "models" / "zones.joblib")     # папку создаёт сам
    assert path.exists() and path.stat().st_size < 5 * 1024 * 1024           # меньше 5 МБ: можно в git

    loaded = zones.load_model(path)
    assert list(loaded.predict(X)) == list(model.predict(X))


def test_load_model_returns_none_when_missing_or_wrong(tmp_path):
    assert zones.load_model(tmp_path / "no.joblib") is None                  # модели нет
    bad = tmp_path / "bad.joblib"
    bad.write_text("это не модель")
    assert zones.load_model(bad) is None                                     # файл битый
    import joblib
    joblib.dump({"model": object(), "features": ["ndvi_07"]}, tmp_path / "old.joblib")
    assert zones.load_model(tmp_path / "old.joblib") is None                 # другой набор признаков


def test_load_model_uses_default_path_and_falls_back_without_model():
    # conftest подменил zones.MODEL_PATH на несуществующий файл: «модели нет», работают правила
    assert not zones.MODEL_PATH.exists()
    assert zones.load_model() is None


# ---------- Зоны по классификатору ----------
def test_find_model_zones_uses_rice_mask_and_nan():
    X, y = fake_dataset()
    model, _ = train_zones.train_and_test(X, y)
    rng = np.random.default_rng(2)

    # поле 4 x 4: левая половина «здоровый рис», правая «соль»
    features = np.zeros((4, 4, N), dtype="float32")
    features[:, :2] = fake_features("healthy", 8, rng).reshape(4, 2, N)
    features[:, 2:] = fake_features("salinity", 8, rng).reshape(4, 2, N)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False                                     # вне поля
    rice = np.ones((4, 4), dtype=bool)
    rice[1, 3] = False                                     # внутри поля, но не рисовый массив
    features[2, 0, 0] = np.nan                             # нет данных в пикселе

    zone_map = zones.find_model_zones(model, features, rice, mask)
    healthy, salinity = zones.ZONE_KEYS.index("healthy") + 1, zones.ZONE_KEYS.index("salinity") + 1
    assert zone_map[0, 0] == 0                             # вне поля
    assert zone_map[1, 1] == healthy and zone_map[3, 3] == salinity
    assert zone_map[1, 3] == zones.NOT_RICE_CODE           # не массив: не рисовое поле
    assert zone_map[2, 0] == zones.NOT_RICE_CODE           # NaN: не классифицируем


def test_point_class_is_from_fixed_list():
    zone_map = np.array([[0, 1, 5, 6]], dtype="uint8")
    assert zones.point_class(zone_map, 0, 0) is None       # вне поля
    assert zones.point_class(zone_map, 0, 1) == "healthy"
    assert zones.point_class(zone_map, 0, 2) is None       # «пусто» нет в списке классов точки
    assert zones.point_class(zone_map, 0, 3) == "not_rice"
    assert zones.point_class(zone_map, 5, 5) is None       # за пределами снимка
    assert zones.POINT_CLASSES == ["healthy", "stress", "salinity", "water", "not_rice"]


# ---------- Признаки точек: пропуски ----------
def fake_field(features):
    """Поле 10 x 10 пикселей в UTM 41N; контур: левая половина (колонки 0-4)."""
    x0, y0 = 500000.0, 4970000.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :5] = True
    return {"features": features, "mask": mask, "transform": from_origin(x0, y0, 10, 10), "crs": "EPSG:32641"}


def point_at(field_data, row, col, cls, field="f1"):
    """Точка (широта, долгота) в центре пикселя (row, col)."""
    t = field_data["transform"]
    lon, lat = warp_transform("EPSG:32641", "EPSG:4326", [t.c + (col + 0.5) * 10], [t.f - (row + 0.5) * 10])
    return {"lat": lat[0], "lon": lon[0], "class": cls, "field": field, "date": "2025-07"}


def test_collect_samples_skips_outside_duplicates_and_conflicts():
    features = np.full((10, 10, N), 0.5, dtype="float32")
    features[0, 0, 3] = np.nan                                  # пиксель без данных
    data = fake_field(features)
    points = [
        point_at(data, 1, 1, "healthy"),                        # хорошая
        point_at(data, 2, 2, "stress"),                         # хорошая
        point_at(data, 2, 2, "stress"),                         # дубль: тот же пиксель и класс
        point_at(data, 3, 3, "healthy"),                        # конфликт: на одном пикселе два класса...
        point_at(data, 3, 3, "salinity"),                       # ...оба убираем
        point_at(data, 4, 8, "water"),                          # вне контура (правая половина)
        point_at(data, 50, 50, "water"),                        # вне снимка совсем
        point_at(data, 0, 0, "healthy"),                        # NaN
        point_at(data, 1, 1, "healthy", field="no_such_field"), # поля нет в словаре
    ]
    X, y, groups, skipped = train_zones.collect_samples(points, {"f1": data})
    assert sorted(y) == ["healthy", "stress"] and X.shape == (2, N) and list(groups) == ["f1", "f1"]
    assert skipped["f1"]["вне контура поля"] == 2               # правая половина и вне снимка
    assert skipped["f1"]["дубль (тот же пиксель)"] == 1
    assert skipped["f1"]["разные классы на одном пикселе"] == 2
    assert skipped["f1"]["нет данных в пикселе (NaN)"] == 1
    assert skipped["no_such_field"]["нет снимков поля"] == 1


# ---------- Отчёт ----------
def make_report(per_class):
    rng = np.random.default_rng(0)
    X = np.vstack([fake_features(c, n, rng) for c, n in per_class.items()])
    y = np.array([c for c, n in per_class.items() for _ in range(n)])
    model, result = train_zones.train_and_test(X, y)
    return result, train_zones.report_text(result, {"f1": {"вне контура поля": 3}}, len(y) + 3, "models/zones.joblib", 480.0)


def test_report_has_all_required_parts_and_honest_warnings():
    result, text = make_report({"healthy": 40, "stress": 40, "salinity": 12, "not_rice": 40})
    assert "Accuracy на отложенной выборке" in text and "достигнута по этой оценке" in text
    assert "| healthy |" in text and "Precision" in text and "F1" in text         # F1 по классам
    assert "Матрица ошибок" in text and "| настоящий" in text
    assert "размечен по виду на снимке" in text and "без проверки на месте" in text   # соль
    assert "Класс **water**: 0 размеченных точек" in text                        # класса нет совсем
    assert "f1: вне контура поля: 3" in text                                     # сколько пропущено
    assert "salinity" in text and "в проверке только 3" in text                  # мало точек в проверке


def test_report_says_so_when_goal_is_missed():
    # два класса с одинаковыми признаками: угадывать нечем, точность около 50%
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (80, N)).astype("float32")
    y = np.array(["healthy", "stress"] * 40)
    model, result = train_zones.train_and_test(X, y)
    text = train_zones.report_text(result, {}, 80, "models/zones.joblib", 100.0)
    assert result["accuracy"] < train_zones.GOAL
    assert "НЕ достигнута" in text and "не подгонялись" in text


def test_report_warns_about_class_with_less_than_5_points():
    _, text = make_report({"healthy": 40, "stress": 40, "not_rice": 40, "salinity": 4})
    assert "Класс **salinity**: всего 4 точек (меньше 5)" in text


# ---------- Проверка по полям и точки в рисовом массиве ----------
def test_leave_one_field_out_and_unseen_classes():
    X, y = fake_dataset(per_class=30)
    groups = np.array(["a", "b", "c"] * 40)                 # три поля вперемешку: классы есть везде
    found = train_zones.leave_one_field_out(X, y, groups)
    assert [r["field"] for r in found] == ["a", "b", "c"] and all(r["n"] == 40 for r in found)
    assert all(r["accuracy"] > 0.9 and r["unseen"] == [] for r in found)

    # соль есть только на поле «c»: при проверке на «c» модель её не видела
    groups = np.where(y == "salinity", "c", np.where(np.arange(len(y)) % 2, "a", "b"))
    found = {r["field"]: r for r in train_zones.leave_one_field_out(X, y, groups)}
    assert found["c"]["unseen"] == ["salinity"] and found["a"]["unseen"] == []


def test_report_shows_points_inside_rice_and_field_check():
    rng = np.random.default_rng(0)
    X = np.vstack([fake_features(c, 40, rng) for c in CENTERS])
    y = np.array([c for c in CENTERS for _ in range(40)])
    X[y == "salinity", -1] = 0.0                            # вся соль вне рисового массива (весной воды не было)
    model, result = train_zones.train_and_test(X, y)
    assert result["inside_rice"]["salinity"] == 0 and result["inside_rice"]["healthy"] == 40
    by_field = [{"field": "f1", "n": 10, "accuracy": 0.6, "unseen": ["salinity"]}]
    text = train_zones.report_text(result, {}, len(y), "models/zones.joblib", 100.0, by_field)
    assert "ни одна из 40 точек не лежит в рисовом массиве" in text
    assert "Проверка по полям" in text and "| f1 | 10 | 60.0% | salinity |" in text
    assert "Ниже 75% на полях: f1" in text and "Но на поле, которого модель не видела" in text
