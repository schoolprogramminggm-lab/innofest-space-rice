"""Тесты деления на зоны (src/zones.py) на искусственных массивах. Интернет не нужен."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import zones

# Четыре явно разных типа пикселей: [NDVI, SI, NDWI]
HEALTHY = [0.90, 0.02, -0.80]   # много зелени
STRESS = [0.60, 0.05, -0.50]    # зелени меньше
SALT = [0.15, 0.20, -0.30]      # мало растений, много соли
WATER = [-0.20, 0.03, 0.30]     # вода


def make_field(blocks):
    """
    Делает поле 10 x 10 из четырёх блоков 5 x 5 (blocks: 4 списка [NDVI, SI, NDWI]
    для левого верхнего, правого верхнего, левого нижнего, правого нижнего).
    Возвращает ndvi, si, ndwi.
    """
    rng = np.random.default_rng(0)   # небольшой шум, как в настоящем снимке
    arrays = np.zeros((3, 10, 10))
    positions = [(0, 0), (0, 5), (5, 0), (5, 5)]
    for (row, col), values in zip(positions, blocks):
        for i in range(3):
            arrays[i, row:row + 5, col:col + 5] = values[i] + rng.normal(0, 0.005, (5, 5))
    return arrays[0], arrays[1], arrays[2]


def test_kmeans_gives_four_zones():
    ndvi, si, ndwi = make_field([HEALTHY, STRESS, SALT, WATER])
    mask = np.ones((10, 10), dtype=bool)
    zone_map, means = zones.find_zones(ndvi, si, ndwi, mask)

    # ровно 4 разные зоны, по 25 пикселей в каждой
    assert sorted(np.unique(zone_map)) == [1, 2, 3, 4]
    for code in [1, 2, 3, 4]:
        assert (zone_map == code).sum() == 25
    assert set(means) == {"healthy", "stress", "salinity", "water"}


def test_names_come_from_means_not_from_position():
    # Меняем блоки местами: номера кластеров у KMeans будут другими,
    # но здоровый рис всё равно должен остаться там, где NDVI высокий
    expected = {id(HEALTHY): "healthy", id(STRESS): "stress", id(SALT): "salinity", id(WATER): "water"}
    for blocks in ([HEALTHY, STRESS, SALT, WATER], [WATER, SALT, STRESS, HEALTHY],
                   [SALT, HEALTHY, WATER, STRESS]):
        ndvi, si, ndwi = make_field(blocks)
        zone_map, _ = zones.find_zones(ndvi, si, ndwi, np.ones((10, 10), dtype=bool))

        # какая зона стоит в центре каждого блока
        centers = [zone_map[2, 2], zone_map[2, 7], zone_map[7, 2], zone_map[7, 7]]
        for block, code in zip(blocks, centers):
            assert zones.ZONE_KEYS[code - 1] == expected[id(block)]


def test_name_clusters_by_means():
    # средние [NDVI, SI, NDWI] четырёх кластеров в «случайном» порядке номеров
    means = [
        [0.15, 0.20, -0.30],   # 0: мало растений, много соли -> salinity
        [0.90, 0.02, -0.80],   # 1: самый высокий NDVI -> healthy
        [-0.20, 0.03, 0.30],   # 2: мало растений, SI низкий -> water
        [0.60, 0.05, -0.50],   # 3: второй по NDVI -> stress
    ]
    assert zones.name_clusters(means) == ["salinity", "healthy", "water", "stress"]


def test_name_clusters_summer_without_water():
    # Летом воды нет: NDWI у всех отрицательный. Голая земля с высоким SI не должна
    # стать «водой» только из-за самого высокого NDWI.
    means = [
        [0.90, 0.01, -0.80],   # healthy
        [0.79, 0.04, -0.70],   # stress
        [0.35, 0.15, -0.38],   # мало растений, высокий SI -> salinity
        [0.30, 0.06, -0.58],   # мало растений, SI ниже -> water
    ]
    assert zones.name_clusters(means) == ["healthy", "stress", "salinity", "water"]


def test_outside_field_and_nan_are_ignored():
    ndvi, si, ndwi = make_field([HEALTHY, STRESS, SALT, WATER])
    mask = np.ones((10, 10), dtype=bool)
    mask[:, 8:] = False        # две колонки справа вне поля
    ndvi[0, 0] = np.nan        # пиксель, где индекс не посчитался

    zone_map, _ = zones.find_zones(ndvi, si, ndwi, mask)
    assert (zone_map[:, 8:] == 0).all()   # вне поля: 0
    assert zone_map[0, 0] == 0            # NaN: 0
    assert (zone_map > 0).sum() == 100 - 20 - 1


def test_zone_areas():
    # 2 пикселя здорового, 1 стресса, 1 вне поля; пиксель 10 x 10 м = 0.01 га
    zone_map = np.array([[1, 1, 2, 0]], dtype="uint8")
    areas = zones.zone_areas(zone_map, pixel_ha=0.01)
    assert areas["healthy"] == {"ha": 0.0, "pct": 66.7}   # 0.02 га округляется до 0.0
    assert areas["stress"]["pct"] == 33.3
    assert areas["salinity"] == {"ha": 0.0, "pct": 0.0}

    # крупнее: 10 000 пикселей = 100 га
    big = np.ones((100, 100), dtype="uint8")
    assert zones.zone_areas(big, 0.01)["healthy"] == {"ha": 100.0, "pct": 100.0}
