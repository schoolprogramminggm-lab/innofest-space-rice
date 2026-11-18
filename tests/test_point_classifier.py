"""
Тесты: analyze_point с классификатором и откат на правила, если модели нет (issue #11).
Интернет не нужен: снимки лежат в кэше во временной папке (помощники из test_point.py).
"""
from datetime import date

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

import download
import point
import zones
from test_point import LAT, LON, make_cached_scene, make_spring, tmp_dirs  # noqa: F401 (tmp_dirs: фикстура)


def put_scenes(months=(5, 7, 8), ndvi_level=4.0):
    """В кэш квадрата вокруг точки кладём снимки нужных месяцев и весну (всё поле было под водой)."""
    prefix = point.cache_prefix(LAT, LON, 500)
    geometry = point.square_geometry(LAT, LON, 500)
    for m in months:
        make_cached_scene(prefix, geometry, download.month_label(m), ndvi_level)
    make_spring(prefix, geometry, flooded_share=1.0)
    return prefix, geometry


def save_constant_model(cls):
    """
    Кладёт в zones.MODEL_PATH модель, которая на любых признаках отвечает один и тот же класс.
    Так видно, что зоны дал именно классификатор, а не пороговые правила.
    """
    model = DummyClassifier(strategy="constant", constant=cls)
    model.fit(np.zeros((2, len(zones.FEATURE_NAMES))), [cls, cls])
    zones.save_model(model, zones.MODEL_PATH)


# ---------- Модели нет: пороговые правила, как раньше ----------
def test_without_model_zones_come_from_rules(tmp_dirs):
    put_scenes()
    assert not zones.MODEL_PATH.exists()
    answer = point.analyze_point(LAT, LON, 500, "2025-07", today=date(2025, 7, 20))
    assert answer["ok"] is True
    result = answer["result"]
    assert result["method"] == "rules"
    assert result["zones"]["healthy"]["pct"] > 90               # густая зелень: по правилам здоровый рис
    assert result["point_class"] == "healthy"                   # класс точки из списка и без модели
    assert "Классификатор не применён" not in answer["message"]["ru"]   # модели не было: пометка не нужна


def test_rules_work_when_only_requested_month_is_cached(tmp_dirs):
    """Модели нет: май и август не нужны, сеть не трогаем (в tmp_dirs она запрещена)."""
    put_scenes(months=(7,))
    assert point.analyze_point(LAT, LON, 500, "2025-07")["result"]["method"] == "rules"


# ---------- Модель есть: зоны даёт классификатор ----------
def test_with_model_zones_and_point_class_come_from_classifier(tmp_dirs):
    put_scenes()
    save_constant_model("salinity")
    answer = point.analyze_point(LAT, LON, 500, "2025-07", today=date(2025, 7, 20))
    assert answer["ok"] is True
    result = answer["result"]
    assert result["method"] == "classifier"
    assert result["zones"]["salinity"]["pct"] > 99              # правила дали бы здоровый рис
    assert result["zones"]["healthy"]["pct"] == 0
    assert result["point_class"] == "salinity"
    assert result["point_class"] in zones.POINT_CLASSES
    assert "Точка: Засоление" in answer["message"]["ru"] and "Нүкте: Тұзданған жер" in answer["message"]["kk"]

    # на карте маркер с этим же классом
    html = (tmp_dirs / "output" / ("map_%s.html" % result["field"])).read_text(encoding="utf-8")
    assert "Точка: Засоление / Тұзданған жер (salinity)" in html


def test_classifier_keeps_not_rice_outside_flooded_area(tmp_dirs):
    """Рисовый массив по весне остаётся прежним: вне его «не рисовое поле», что бы ни думала модель."""
    prefix, geometry = put_scenes()
    point.cache_path(prefix, "spring_orbits").unlink()
    make_spring(prefix, geometry, flooded_share=0.5)            # затоплена левая половина
    save_constant_model("healthy")
    result = point.analyze_point(LAT, LON, 500, "2025-07")["result"]
    assert result["method"] == "classifier"
    assert result["zones"]["healthy"]["pct"] == pytest.approx(50, abs=5)
    assert result["zones"]["not_rice"]["pct"] == pytest.approx(50, abs=5)


def test_broken_model_file_falls_back_to_rules(tmp_dirs):
    put_scenes()
    zones.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    zones.MODEL_PATH.write_text("не модель")
    assert point.analyze_point(LAT, LON, 500, "2025-07")["result"]["method"] == "rules"


# ---------- Модель есть, но снимков за три месяца не хватает: правила и честная пометка ----------
def test_model_but_missing_month_falls_back_to_rules_with_note(tmp_dirs, monkeypatch):
    put_scenes(months=(7,))                                     # мая и августа в кэше нет
    save_constant_model("salinity")
    real_fetch = point.fetch_tif
    monkeypatch.setattr(point, "fetch_tif",                     # за май «нет безоблачного снимка»
                        lambda geometry, month, tif, state: "no_image" if month == 5 else real_fetch(geometry, month, tif, state))
    answer = point.analyze_point(LAT, LON, 500, "2025-07")
    assert answer["ok"] is True
    assert answer["result"]["method"] == "rules"
    assert answer["result"]["zones"]["salinity"]["pct"] == 0    # классификатор не использован
    assert "Классификатор не применён" in answer["message"]["ru"]
    assert "Жіктегіш қолданылмады" in answer["message"]["kk"]


def test_no_spring_images_uses_rules_even_with_model(tmp_dirs):
    """Без весенних снимков признака затопления нет: классификатор не применяем, как и раньше «без проверки затопления»."""
    prefix = point.cache_prefix(LAT, LON, 500)
    geometry = point.square_geometry(LAT, LON, 500)
    for label in ("2025-07", "2025-08"):
        make_cached_scene(prefix, geometry, label, ndvi_level=4.0)
    make_spring(prefix, geometry, flooded_share=0.0, images=0)
    save_constant_model("salinity")
    answer = point.analyze_point(LAT, LON, 500, "2025-07")
    assert answer["result"]["method"] == "rules"
    assert "Без проверки затопления" in answer["message"]["ru"]


def test_geojson_has_no_point_class(tmp_dirs):
    """Для своего контура точки нет: point_class равен null."""
    prefix, geometry = put_scenes()
    answer = point.analyze_geometry("geojson_x", geometry, "2025-07", prefix, today=date(2025, 7, 20))
    assert answer["ok"] is True and answer["result"]["point_class"] is None
