"""Тесты point.py: любое поле по точке. Интернет и ключи не нужны."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import download
import point

LAT, LON = 44.9716, 64.5267      # поле 3 в Жалагашском районе
KEYS = {"module", "version", "ok", "result", "message", "error"}


def to_utm(lons, lats):
    """Градусы -> метры (UTM 41N), чтобы мерить квадрат линейкой."""
    return warp_transform("EPSG:4326", "EPSG:%d" % download.utm_epsg(LON), lons, lats)


# ---------- Квадрат вокруг точки ----------
@pytest.mark.parametrize("size", [200, 500, 1000])
def test_square_area_and_corners(size):
    geometry = point.square_geometry(LAT, LON, size)
    assert geometry["type"] == "Polygon"
    ring = geometry["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]        # 4 угла и замыкающая точка

    xs, ys = to_utm([c[0] for c in ring], [c[1] for c in ring])
    # площадь по формуле Гаусса (шнурок): должна быть почти size x size
    area = 0.5 * abs(sum(xs[i] * ys[i + 1] - xs[i + 1] * ys[i] for i in range(4)))
    assert area == pytest.approx(size * size, rel=0.001)

    # стороны равны size, а углы прямые: скалярное произведение соседних сторон равно 0
    sides = [(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(4)]
    for i in range(4):
        assert np.hypot(*sides[i]) == pytest.approx(size, rel=0.001)
        nxt = sides[(i + 1) % 4]
        assert abs(sides[i][0] * nxt[0] + sides[i][1] * nxt[1]) < size * size * 0.001

    # центр квадрата там, где точка
    cx, cy = to_utm([LON], [LAT])
    assert np.mean(xs[:4]) == pytest.approx(cx[0], abs=0.5)
    assert np.mean(ys[:4]) == pytest.approx(cy[0], abs=0.5)


def test_square_goes_into_download_request():
    """Квадрат подходит download.py: рамка запроса чуть больше 500 м, пикселей около 50 x 50."""
    body = download.build_request(point.square_geometry(LAT, LON, 500), 7)
    assert 50 <= body["output"]["width"] <= 54
    assert 50 <= body["output"]["height"] <= 54


# ---------- Разбор --point и --month ----------
def test_parse_point():
    assert point.parse_point("44.9716,64.5267") == (44.9716, 64.5267)
    assert point.parse_point(" 44.9716 , 64.5267 ") == (44.9716, 64.5267)


@pytest.mark.parametrize("text", ["", "44.9716", "44.9716,64.5267,1", "abc,def", "95,64", "44,200", "-10,64"])
def test_parse_point_rejects_bad_text(text):
    with pytest.raises(ValueError):
        point.parse_point(text)


def test_parse_month():
    assert point.parse_month("2025-07") == 7
    assert point.parse_month("2025-05") == 5
    for bad in ["07", "2025-7", "2025-06", "2024-07", "июль"]:
        with pytest.raises(ValueError):
            point.parse_month(bad)


# ---------- Кэш ----------
def test_cache_key_and_path():
    assert point.cache_key(LAT, LON, 500, "2025-07") == "44.9716_64.5267_500_2025-07"
    # лишние знаки отбрасываются: рядом стоящая точка (около 5 м) даёт тот же ключ
    assert point.cache_key(44.97162, 64.52668, 500, "2025-07") == "44.9716_64.5267_500_2025-07"
    # другой размер, месяц или место дают другой ключ
    assert point.cache_key(LAT, LON, 800, "2025-07") != point.cache_key(LAT, LON, 500, "2025-07")
    assert point.cache_key(LAT, LON, 500, "2025-08") != point.cache_key(LAT, LON, 500, "2025-07")
    assert point.cache_key(LAT + 0.01, LON, 500, "2025-07") != point.cache_key(LAT, LON, 500, "2025-07")

    path = point.cache_path(point.cache_prefix(LAT, LON, 500), "2025-07")
    assert path.name == "44.9716_64.5267_500_2025-07.tif"
    assert path.parent.name == "cache" and path.parent.parent.name == "data"


# ---------- Честные сообщения ----------
def test_no_rice_message_on_artificial_array():
    mask = np.ones((10, 10), dtype=bool)
    bare = np.full((10, 10), 0.1, dtype="float32")       # голая земля: NDVI 0.1
    answer = point.rice_check(bare, mask)
    assert set(answer) == KEYS
    assert answer["ok"] is False and answer["result"] is None
    assert "нет посевов риса" in answer["message"]["ru"]
    assert answer["message"]["kk"].strip()
    assert answer["error"]

    green = np.full((10, 10), 0.7, dtype="float32")      # рис в росте
    assert point.rice_check(green, mask) is None


def test_rice_check_looks_only_inside_mask_and_skips_nan():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5] = True
    ndvi = np.full((10, 10), 0.8, dtype="float32")
    ndvi[5:] = 0.0                                       # вне контура пусто: не должно мешать
    ndvi[0, 0] = np.nan
    assert point.rice_check(ndvi, mask) is None

    ndvi[:5] = 0.1                                       # а внутри контура пусто
    assert point.rice_check(ndvi, mask) is not None


def test_season_note():
    note = point.season_note(date(2026, 11, 17))         # поля убраны
    assert "убраны" in note["ru"] and "2025" in note["ru"] and note["kk"].strip()
    assert point.season_note(date(2025, 7, 10)) is None


def test_no_image_answer_has_both_languages():
    answer = point.no_image_answer("2025-05")
    assert set(answer) == KEYS and answer["ok"] is False
    assert "безоблачного" in answer["message"]["ru"] and "2025-05" in answer["message"]["kk"]


@pytest.mark.parametrize("kwargs", [
    {"lat": 95, "lon": LON, "size": 500, "month": "2025-07"},   # такой широты нет
    {"lat": LAT, "lon": LON, "size": 50, "month": "2025-07"},   # квадрат слишком маленький
    {"lat": LAT, "lon": LON, "size": 99999, "month": "2025-07"},
    {"lat": LAT, "lon": LON, "size": 500, "month": "2025-06"},  # июнь не качаем
])
def test_bad_input_gives_ok_false_without_network(kwargs, monkeypatch):
    # если функция вдруг полезет в сеть, тест упадёт
    monkeypatch.setattr(download, "get_token", lambda: pytest.fail("лезет в сеть"))
    answer = point.analyze_point(**kwargs)
    assert set(answer) == KEYS and answer["ok"] is False
    assert answer["message"]["kk"].strip() and answer["message"]["ru"].strip()


# ---------- Весь конвейер на кэшированном снимке (сети нет) ----------
def make_cached_scene(prefix, geometry, month_label, ndvi_level):
    """Кладёт в кэш искусственный снимок поверх квадрата. ndvi_level: красный/ИК канал."""
    bbox, epsg, width, height = download.field_bbox(geometry)
    rng = np.random.default_rng(1)
    data = rng.uniform(0.03, 0.08, (5, height, width)).astype("float32")
    data[2] = 0.05                                       # B04 красный
    data[3] = 0.05 + ndvi_level * rng.uniform(0.2, 1.0, (height, width)) * 0.1 + ndvi_level * 0.2
    tif = point.cache_path(prefix, month_label)
    tif.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(bbox[0], bbox[3], 10, 10)
    with rasterio.open(tif, "w", driver="GTiff", height=height, width=width, count=5,
                       dtype="float32", crs="EPSG:%d" % epsg, transform=transform) as dst:
        dst.write(data)


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    """Кэш и результаты кладём во временную папку, сеть запрещаем."""
    monkeypatch.setattr(point, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(point, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(download, "get_token", lambda: pytest.fail("лезет в сеть: кэш не сработал"))
    return tmp_path


def test_analyze_point_uses_cache_and_returns_agroqyzyl(tmp_dirs):
    prefix = point.cache_prefix(LAT, LON, 500)
    geometry = point.square_geometry(LAT, LON, 500)
    make_cached_scene(prefix, geometry, "2025-07", ndvi_level=4.0)   # густая зелень

    answer = point.analyze_point(LAT, LON, 500, "2025-07", today=date(2025, 7, 20))
    assert set(answer) == KEYS
    assert answer["ok"] is True and answer["error"] is None
    result = answer["result"]
    assert result["area_ha"] == pytest.approx(25, abs=0.7)           # 500 x 500 м = 25 га
    assert sum(z["ha"] for z in result["zones"].values()) == pytest.approx(result["area_ha"], abs=0.3)
    assert (tmp_dirs / "output" / ("map_%s.html" % result["field"])).exists()
    assert (tmp_dirs / "output" / ("ndvi_%s_2025-07.png" % result["field"])).exists()
    assert "убраны" not in answer["message"]["ru"]                   # в июле поля не убраны

    # осенью к тексту добавляется пояснение про сезон 2025
    late = point.analyze_point(LAT, LON, 500, "2025-07", today=date(2026, 11, 17))
    assert "сезон 2025" in late["message"]["ru"]


def test_analyze_point_no_rice_on_bare_land(tmp_dirs):
    prefix = point.cache_prefix(LAT, LON, 500)
    geometry = point.square_geometry(LAT, LON, 500)
    make_cached_scene(prefix, geometry, "2025-07", ndvi_level=0.0)   # NIR = Red: NDVI около 0

    answer = point.analyze_point(LAT, LON, 500, "2025-07")
    assert answer["ok"] is False
    assert "нет посевов риса" in answer["message"]["ru"]
