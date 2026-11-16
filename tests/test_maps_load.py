"""Тесты чтения снимка по контуру (src/load.py) и карты (src/maps.py). Интернет не нужен."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import download
import load
import maps

# Точка около Жалагаша в градусах и в метрах (UTM 41N)
LON, LAT = 64.52, 44.98
_x, _y = warp_transform("EPSG:4326", "EPSG:32641", [LON], [LAT])
X0, Y0 = _x[0], _y[0]
CRS = rasterio.crs.CRS.from_epsg(32641)


def make_tif(path):
    """Снимок 10 x 10 пикселей по 10 м, 5 каналов, левый верхний угол в точке (X0, Y0)."""
    transform = from_origin(X0, Y0, 10, 10)
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=5,
                       dtype="float32", crs="EPSG:32641", transform=transform) as dst:
        dst.write(np.full((5, 10, 10), 0.2, dtype="float32"))
    return transform


def test_load_scene_clips_to_field(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download, "FIELDS_DIR", tmp_path)
    make_tif(tmp_path / "field_9_2025-07.tif")

    # контур поля: левая половина снимка (x от X0 до X0+50 м). Углы переводим в градусы.
    xs = [X0, X0 + 50, X0 + 50, X0, X0]
    ys = [Y0, Y0, Y0 - 100, Y0 - 100, Y0]
    lons, lats = warp_transform("EPSG:32641", "EPSG:4326", xs, ys)
    polygon = {"type": "Polygon", "coordinates": [[list(p) for p in zip(lons, lats)]]}
    (tmp_path / "field_9.geojson").write_text(json.dumps(polygon), encoding="utf-8")

    scene = load.load_scene("field_9", 7)
    assert set(scene["bands"]) == {"B02", "B03", "B04", "B08", "B11"}
    assert scene["bands"]["B04"].shape == (10, 10)
    assert np.isclose(scene["pixel_ha"], 0.01)          # 10 x 10 м = 0.01 га
    # внутри поля левая половина: около 50 пикселей (на краю возможна разница в 1 колонку)
    assert 45 <= scene["mask"].sum() <= 55
    assert scene["mask"][:, :4].all()                   # левая часть точно внутри
    assert not scene["mask"][:, 6:].any()               # правая часть точно снаружи


def test_zones_to_rgba_colors():
    zone_map = np.array([[0, 1], [3, 4]], dtype="uint8")
    rgba = maps.zones_to_rgba(zone_map)
    assert rgba[0, 0, 3] == 0                           # вне поля: прозрачно
    assert tuple(rgba[0, 1]) == (*maps.ZONE_COLORS[0], 255)
    assert tuple(rgba[1, 0]) == (*maps.ZONE_COLORS[2], 255)


def test_zones_to_latlon_bounds(tmp_path):
    transform = make_tif(tmp_path / "a.tif")
    zone_map = np.ones((10, 10), dtype="uint8")
    result, bounds = maps.zones_to_latlon(zone_map, transform, CRS)
    (south, west), (north, east) = bounds
    # порядок в bounds: сначала широта, потом долгота
    assert abs(south - (LAT - 0.001)) < 0.001 and abs(north - LAT) < 0.001
    assert abs(west - LON) < 0.001 and abs(east - LON) < 0.002
    assert result.max() == 1


def test_map_has_legend_in_both_languages(tmp_path):
    transform = make_tif(tmp_path / "a.tif")
    zone_map = np.random.default_rng(1).integers(1, 5, (10, 10)).astype("uint8")
    polygon = {"type": "Polygon", "coordinates": [[[LON, LAT], [LON + 0.001, LAT],
                                                    [LON + 0.001, LAT - 0.001], [LON, LAT]]]}
    out = tmp_path / "map_field_9.html"
    scenes = {5: (zone_map, transform, CRS), 7: (zone_map, transform, CRS)}
    maps.save_map("field_9", polygon, scenes, out)

    html = out.read_text(encoding="utf-8")
    for ru, kk in maps.ZONE_LABELS:     # все четыре зоны на русском и на казахском
        assert ru in html and kk in html
    # названия слоёв folium пишет в JavaScript с экранированием (И...), сравниваем так же
    assert json.dumps("Июль / Шілде 2025")[1:-1] in html
    assert json.dumps("Май / Мамыр 2025")[1:-1] in html
    assert "field_9" in html            # подпись контура
