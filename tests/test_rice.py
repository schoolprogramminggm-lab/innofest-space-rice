"""Тесты маски риса (src/rice.py, src/spring.py, зоны по маске) на искусственных массивах. Сети нет."""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import download
import rice
import spring
import zones

SHAPE = (10, 10)
FIELD = np.ones(SHAPE, dtype=bool)          # всё внутри поля


def full(value):
    return np.full(SHAPE, value, dtype="float32")


def test_green_grass_without_spring_water_is_not_rice():
    """Трава: летом NDVI высокий, но весной воды не было -> не рис."""
    summer = rice.summer_ndvi(full(0.85), full(0.80))
    flood = full(-0.3)                                    # весной сухо (NDWI меньше 0)
    pixels = rice.rice_mask(summer, flood, FIELD)
    assert not pixels.any()
    assert rice.rice_share(pixels, FIELD) == 0.0
    assert not rice.is_rice_field(0.0)


def test_flooded_in_spring_and_green_in_summer_is_rice():
    pixels = rice.rice_mask(full(0.85), full(0.25), FIELD)
    assert pixels.all()
    assert rice.rice_share(pixels, FIELD) == 1.0
    assert rice.is_rice_field(1.0)


def test_flooded_but_not_green_is_not_rice():
    """Вода была, но рис не взошёл (соль, пустая земля): летом NDVI низкий -> не рис."""
    assert not rice.rice_mask(full(0.2), full(0.25), FIELD).any()


def test_july_or_august_is_enough():
    july, august = full(0.3), full(0.8)                   # в июле ещё низко, в августе высоко
    assert rice.summer_ndvi(july, august)[0, 0] == pytest.approx(0.8)
    # NaN в одном месяце (облако) не мешает другому
    july[:] = np.nan
    assert rice.summer_ndvi(july, august)[0, 0] == pytest.approx(0.8)
    assert rice.rice_mask(rice.summer_ndvi(july, august), full(0.1), FIELD).all()


def test_less_than_30_percent_rice_is_not_a_rice_field():
    flood = full(-0.3)
    flood[:, :2] = 0.3                                    # 2 колонки из 10 = 20% были затоплены
    pixels = rice.rice_mask(full(0.9), flood, FIELD)
    share = rice.rice_share(pixels, FIELD)
    assert share == pytest.approx(0.2)
    assert not rice.is_rice_field(share)

    flood[:, :3] = 0.3                                    # 30%: ровно на пороге, уже рис
    assert rice.is_rice_field(rice.rice_share(rice.rice_mask(full(0.9), flood, FIELD), FIELD))


def test_only_inside_field_and_nan_never_rice():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:5] = True                                       # поле это верхняя половина
    summer = full(0.9)
    summer[0, 0] = np.nan
    pixels = rice.rice_mask(summer, full(0.3), mask)
    assert pixels.sum() == 49                             # 50 пикселей поля минус один NaN
    assert not pixels[5:].any()
    assert rice.rice_share(pixels, mask) == pytest.approx(0.98)


def test_no_spring_images_falls_back_to_summer_only():
    pixels = rice.rice_mask(full(0.9), None, FIELD)       # flood=None: весенних снимков нет
    assert pixels.all()


# ---------- Зоны по маске ----------
def test_rice_zones_only_on_rice_pixels():
    ndvi = full(0.9)
    ndvi[:, 5:] = 0.7                                     # правая половина слабее
    si, ndwi = full(0.05), full(-0.7)
    rice_pixels = np.zeros(SHAPE, dtype=bool)
    rice_pixels[:, :8] = True                             # риса 80%, две колонки справа не рис
    zone_map, means = zones.find_rice_zones(ndvi, si, ndwi, rice_pixels, FIELD)

    assert (zone_map[:, 8:] == zones.NOT_RICE_CODE).all()
    assert (zone_map[:, :5] == 1).all()                   # высокий NDVI: здоровый рис
    assert (zone_map[:, 5:8] == 2).all()                  # ниже: стресс
    assert means["healthy"][0] > means["stress"][0]

    areas = zones.zone_areas(zone_map, 0.01)
    assert areas["not_rice"]["pct"] == 20.0
    assert areas["healthy"]["pct"] + areas["stress"]["pct"] == 80.0


def test_rice_zones_without_rice_everything_is_not_rice_and_outside_stays_zero():
    mask = FIELD.copy()
    mask[:, 8:] = False
    zone_map, means = zones.find_rice_zones(full(0.1), full(0.1), full(-0.1), np.zeros(SHAPE, bool), mask)
    assert means == {}
    assert (zone_map[:, :8] == zones.NOT_RICE_CODE).all()
    assert (zone_map[:, 8:] == 0).all()                   # вне поля по-прежнему 0


# ---------- Весенний файл ----------
def test_spring_request_covers_20_april_to_20_june():
    geometry = {"type": "Polygon", "coordinates": [[[64.52, 44.97], [64.53, 44.97],
                                                    [64.53, 44.98], [64.52, 44.97]]]}
    body = spring.build_request(geometry)
    window = body["input"]["data"][0]["dataFilter"]["timeRange"]
    assert window["from"].startswith("2025-04-20") and window["to"].startswith("2025-06-20")
    assert body["input"]["data"][0]["dataFilter"]["maxCloudCoverage"] == download.MAX_CLOUDS
    assert "SCL" in body["evalscript"] and "ORBIT" in body["evalscript"]   # облака убираем по SCL


def write_spring(path, ndwi, count):
    transform = from_origin(500000, 4970000, 10, 10)
    with rasterio.open(path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=2,
                       dtype="float32", crs="EPSG:32641", transform=transform) as dst:
        dst.write(np.stack([ndwi, count]))


def test_spring_read(tmp_path):
    ndwi, count = full(0.2), full(3)
    count[0, 0] = 0                                       # в одном пикселе все снимки были в облаках
    write_spring(tmp_path / "a.tif", ndwi, count)
    flood, images = spring.read(tmp_path / "a.tif", SHAPE)
    assert images == 3
    assert np.isnan(flood[0, 0]) and flood[1, 1] == pytest.approx(0.2)
    with pytest.raises(ValueError):                       # снимок другого размера не склеиваем молча
        spring.read(tmp_path / "a.tif", (5, 5))


def test_spring_without_images_returns_none(tmp_path):
    write_spring(tmp_path / "a.tif", full(-2), full(0))
    # файл уже в кэше, поэтому в сеть (get_token) функция не идёт
    flood, images = spring.flood({}, tmp_path / "a.tif", SHAPE, lambda: pytest.fail("лезет в сеть"))
    assert flood is None and images == 0
