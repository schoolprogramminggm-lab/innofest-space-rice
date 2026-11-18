"""Тесты рисового массива (src/rice.py, src/spring.py) и зон внутри него на искусственных массивах. Сети нет."""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import download
import indices
import rice
import spring
import zones

SHAPE = (10, 10)
FIELD = np.ones(SHAPE, dtype=bool)          # всё внутри поля
HEALTHY, STRESS, SALT, WATER, EMPTY, NOT_RICE = [zones.ZONE_KEYS.index(k) + 1 for k in
                                                  ("healthy", "stress", "salinity", "water", "empty", "not_rice")]


def full(value):
    return np.full(SHAPE, value, dtype="float32")


def classify(ndvi, si, ndwi, flooded):
    """Зоны для поля 10 x 10 из одинаковых значений во всех пикселях."""
    pixels = rice.rice_array(flooded, FIELD)
    zone_map, _ = zones.find_rice_zones(full(ndvi), full(si), full(ndwi), pixels, FIELD)
    return zone_map


# ---------- Признак затопления (Xiao et al., 2005): LSWI + 0.05 >= NDVI ----------
def test_is_flooded_formula():
    # водная гладь: NDVI около 0, LSWI заметно выше
    assert rice.is_flooded(0.05, 0.10)
    # граница: LSWI + 0.05 ровно равно NDVI: затоплен (знак >=)
    assert rice.is_flooded(0.25, 0.20)
    # сухая земля: LSWI отрицательный, NDVI выше
    assert not rice.is_flooded(0.10, -0.12)
    # густая зелень: NDVI намного больше LSWI
    assert not rice.is_flooded(0.85, 0.20)
    # нет данных
    assert not rice.is_flooded(np.nan, 0.1)


def test_lswi_from_bands_then_flooded():
    # поле с водой: B8 мал, B11 ещё меньше (вода поглощает SWIR) -> LSWI > 0
    water = indices.lswi(np.array([0.05]), np.array([0.02]))
    ndvi_water = indices.ndvi(np.array([0.06]), np.array([0.05]))
    assert water[0] == pytest.approx(0.4286, abs=1e-3) and rice.is_flooded(ndvi_water, water)[0]
    # сухая голая земля: B11 больше B8 -> LSWI < 0, NDVI выше -> не затоплена
    dry = indices.lswi(np.array([0.30]), np.array([0.40]))
    ndvi_dry = indices.ndvi(np.array([0.25]), np.array([0.30]))
    assert dry[0] < 0 and not rice.is_flooded(ndvi_dry, dry)[0]


# ---------- Рисовый массив и доля ----------
def test_not_flooded_is_not_rice_even_if_green():
    """Трава: летом зелёная, но весной не затоплялась -> не рисовое поле."""
    pixels = rice.rice_array(np.zeros(SHAPE, dtype=bool), FIELD)
    assert not pixels.any()
    assert rice.rice_share(pixels, FIELD) == 0.0
    assert not rice.is_rice_field(0.0)
    zone_map = classify(ndvi=0.85, si=0.02, ndwi=-0.7, flooded=np.zeros(SHAPE, dtype=bool))
    assert (zone_map == NOT_RICE).all()


def test_flooded_in_spring_is_rice_array():
    pixels = rice.rice_array(np.ones(SHAPE, dtype=bool), FIELD)
    assert pixels.all() and rice.rice_share(pixels, FIELD) == 1.0 and rice.is_rice_field(1.0)


def test_less_than_30_percent_is_not_a_rice_field():
    flooded = np.zeros(SHAPE, dtype=bool)
    flooded[:, :2] = True                                 # 2 колонки из 10 = 20%
    share = rice.rice_share(rice.rice_array(flooded, FIELD), FIELD)
    assert share == pytest.approx(0.2) and not rice.is_rice_field(share)
    flooded[:, :3] = True                                 # 30%: ровно на пороге, уже рис
    assert rice.is_rice_field(rice.rice_share(rice.rice_array(flooded, FIELD), FIELD))


def test_rice_array_only_inside_field():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:5] = True
    pixels = rice.rice_array(np.ones(SHAPE, dtype=bool), mask)
    assert pixels.sum() == 50 and rice.rice_share(pixels, mask) == 1.0


def test_no_spring_images_falls_back_to_summer_ndvi():
    summer = rice.summer_ndvi(full(0.3), full(0.8))       # июль 0.3, август 0.8: берём лучший
    assert summer[0, 0] == pytest.approx(0.8)
    assert rice.rice_array(None, FIELD, summer).all()     # flooded=None: весны нет
    assert not rice.rice_array(None, FIELD, full(0.5)).any()


# ---------- Зоны внутри массива ----------
def test_flooded_and_green_summer_is_healthy_rice():
    assert (classify(0.85, 0.02, -0.7, np.ones(SHAPE, dtype=bool)) == HEALTHY).all()


def test_flooded_and_not_green_with_high_si_is_salinity():
    zone_map = classify(ndvi=0.15, si=0.20, ndwi=-0.3, flooded=np.ones(SHAPE, dtype=bool))
    assert (zone_map == SALT).all()


def test_zone_rules_inside_array():
    ones = np.ones(SHAPE, dtype=bool)
    assert (classify(0.45, 0.03, -0.5, ones) == STRESS).all()     # 0.3-0.6: стресс
    assert (classify(0.60, 0.03, -0.5, ones) == STRESS).all()     # ровно 0.6 ещё стресс (здоровый: выше)
    assert (classify(0.10, 0.04, 0.30, ones) == WATER).all()      # мало растений, SI низкий, вода летом
    assert (classify(0.10, 0.04, -0.30, ones) == EMPTY).all()     # мало растений, остальное: пусто
    assert (classify(0.10, 0.20, 0.30, ones) == SALT).all()       # высокий SI важнее воды (как в правиле)


def test_zones_mixed_field_and_areas():
    """Левая половина: затоплена (внутри массива, 4 колонки здоровый рис, 1 колонка соль), правая: не рис."""
    flooded = np.zeros(SHAPE, dtype=bool)
    flooded[:, :5] = True
    ndvi, si, ndwi = full(0.85), full(0.02), full(-0.7)
    ndvi[:, 4] = 0.1
    si[:, 4] = 0.2
    zone_map, means = zones.find_rice_zones(ndvi, si, ndwi, rice.rice_array(flooded, FIELD), FIELD)
    assert (zone_map[:, :4] == HEALTHY).all() and (zone_map[:, 4] == SALT).all()
    assert (zone_map[:, 5:] == NOT_RICE).all()
    assert means["healthy"][0] == pytest.approx(0.85) and means["salinity"][1] == pytest.approx(0.2)
    areas = zones.zone_areas(zone_map, 0.01)
    assert (areas["healthy"]["pct"], areas["salinity"]["pct"], areas["not_rice"]["pct"]) == (40.0, 10.0, 50.0)
    assert sum(a["pct"] for a in areas.values()) == pytest.approx(100.0)


def test_outside_field_stays_zero_and_nan_is_not_classified():
    mask = FIELD.copy()
    mask[:, 8:] = False
    ndvi = full(0.85)
    ndvi[0, 0] = np.nan
    zone_map, _ = zones.find_rice_zones(ndvi, full(0.02), full(-0.7), np.ones(SHAPE, dtype=bool), mask)
    assert (zone_map[:, 8:] == 0).all()                   # вне поля 0
    assert zone_map[0, 0] == NOT_RICE                     # нет данных: не считаем рисом
    assert (zone_map == HEALTHY).sum() == 79


# ---------- Весенний файл: по каналу на пролёт, облачность считается над полем ----------
def test_spring_request_covers_20_april_to_20_june():
    geometry = {"type": "Polygon", "coordinates": [[[64.52, 44.97], [64.53, 44.97],
                                                    [64.53, 44.98], [64.52, 44.97]]]}
    body = spring.build_request(geometry)
    data_filter = body["input"]["data"][0]["dataFilter"]
    assert data_filter["timeRange"]["from"].startswith("2025-04-20")
    assert data_filter["timeRange"]["to"].startswith("2025-06-20")
    # облачность сцены только грубый предфильтр (80%), а не 20%
    assert data_filter["maxCloudCoverage"] == rice.SCENE_CLOUD_PREFILTER == 80
    script = body["evalscript"]
    assert "SCL" in script and "ORBIT" in script         # облака и тени убираем по SCL
    assert "B11" in script and "lswi + 0.05 >= ndvi" in script   # правило Xiao 2005


def write_spring(path, bands, shape=SHAPE):
    """Весенний файл: bands это список массивов-пролётов со значениями 0 сухо, 1 затоплено, 2 облако."""
    with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1], count=len(bands),
                       dtype="uint8", crs="EPSG:32641", transform=from_origin(500000, 4970000, 10, 10)) as dst:
        dst.write(np.stack(bands).astype("uint8"))


def orbit(flooded_cols=0, cloudy_cols=0):
    """Пролёт: левые cloudy_cols колонок в облаке, дальше flooded_cols колонок затоплено, остальное сухо."""
    band = np.full(SHAPE, spring.DRY, dtype="uint8")
    band[:, cloudy_cols:cloudy_cols + flooded_cols] = spring.FLOODED
    band[:, :cloudy_cols] = spring.BAD
    return band


def test_spring_cloud_is_counted_over_the_field_not_the_scene(tmp_path):
    clear = orbit(flooded_cols=4)                        # над полем чисто, затоплено 40%
    cloudy = orbit(flooded_cols=5, cloudy_cols=5)        # облако на 50% поля: снимок не берём
    edge = orbit(flooded_cols=0, cloudy_cols=1)          # облако на 10% поля (< 20%): берём
    none = np.full(SHAPE, spring.NO_ORBIT, dtype="uint8")  # пустой канал: пролёта нет
    write_spring(tmp_path / "a.tif", [clear, cloudy, edge, none])

    flooded, images = spring.read(tmp_path / "a.tif", SHAPE, FIELD)
    assert images == 2                                   # clear и edge; cloudy и пустой канал не считаются
    assert flooded[:, :4].all() and not flooded[:, 4:].any()   # затопленное на облачном снимке не учтено

    stats = spring.orbit_stats(tmp_path / "a.tif", SHAPE, FIELD)
    assert [s["used"] for s in stats] == [True, False, True]
    assert stats[1]["cloud"] == pytest.approx(0.5)


def test_cloud_outside_the_field_does_not_matter(tmp_path):
    band = orbit(flooded_cols=3)
    band[:, 5:] = spring.BAD                             # вся правая половина в облаках...
    mask = FIELD.copy()
    mask[:, 5:] = False                                  # ...но поле это только левая половина
    write_spring(tmp_path / "a.tif", [band])
    flooded, images = spring.read(tmp_path / "a.tif", SHAPE, mask)
    assert images == 1 and flooded[:, :3].all()


def test_cloud_threshold_is_20_percent_of_field(tmp_path):
    write_spring(tmp_path / "a.tif", [orbit(cloudy_cols=2)])    # ровно 20%: уже пропускаем (нужно меньше 20%)
    assert spring.read(tmp_path / "a.tif", SHAPE, FIELD)[1] == 0
    write_spring(tmp_path / "b.tif", [orbit(cloudy_cols=1)])    # 10%: берём
    assert spring.read(tmp_path / "b.tif", SHAPE, FIELD)[1] == 1


def test_spring_read_checks_shape(tmp_path):
    write_spring(tmp_path / "a.tif", [orbit()])
    with pytest.raises(ValueError):                       # снимок другого размера не склеиваем молча
        spring.read(tmp_path / "a.tif", (5, 5), np.ones((5, 5), dtype=bool))


def test_spring_without_suitable_images_returns_none(tmp_path):
    """Все снимки в облаках над полем: подходящих нет -> (None, 0), включится запасной вариант."""
    write_spring(tmp_path / "a.tif", [orbit(cloudy_cols=10), orbit(cloudy_cols=6)])
    # файл уже в кэше, поэтому в сеть (get_token) функция не идёт
    got, images = spring.flood({}, tmp_path / "a.tif", SHAPE, FIELD, lambda: pytest.fail("лезет в сеть"))
    assert got is None and images == 0
