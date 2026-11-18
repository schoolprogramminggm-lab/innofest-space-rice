"""Тесты инструмента разметки (src/label_tool.py): HTML собирается без сети и без настоящих снимков."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import download
import label_tool

LON, LAT = 64.52, 44.98
_x, _y = warp_transform("EPSG:4326", "EPSG:32641", [LON], [LAT])
X0, Y0 = _x[0], _y[0]


def make_field(tmp_path):
    """Снимок 10 x 10 пикселей и контур поля поверх него в папке tmp_path."""
    with rasterio.open(tmp_path / "field_9_2025-07.tif", "w", driver="GTiff", height=10, width=10,
                       count=5, dtype="float32", crs="EPSG:32641",
                       transform=from_origin(X0, Y0, 10, 10)) as dst:
        dst.write(np.full((5, 10, 10), 0.15, dtype="float32"))
    xs = [X0, X0 + 100, X0 + 100, X0, X0]
    ys = [Y0, Y0, Y0 - 100, Y0 - 100, Y0]
    lons, lats = warp_transform("EPSG:32641", "EPSG:4326", xs, ys)
    polygon = {"type": "Polygon", "coordinates": [[list(p) for p in zip(lons, lats)]]}
    (tmp_path / "field_9.geojson").write_text(json.dumps(polygon), encoding="utf-8")


def test_label_map_has_classes_button_and_csv_format(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download, "FIELDS_DIR", tmp_path)
    make_field(tmp_path)

    out = label_tool.save_label_map("field_9", "2025-07", tmp_path / "label_field_9.html")
    html = out.read_text(encoding="utf-8")

    # пять классов: подпись в панели и ключ в CSV (ключи как в zones.py)
    for key, label, _ in label_tool.CLASSES:
        assert 'value="%s"' % key in html and label in html
    assert [k for k, _, _ in label_tool.CLASSES] == ["healthy", "stress", "salinity", "water", "not_rice"]
    assert "Скачать CSV" in html and "points.csv" in html
    assert "lat,lon,class,field,date" in html                  # заголовок CSV из задания
    assert '"field_9"' in html and '"2025-07"' in html         # поле и дата попадают в каждую строку
    assert "Sentinel-2, 2025-07" in html                       # слой с настоящим снимком


def test_label_map_rejects_bad_month(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download, "FIELDS_DIR", tmp_path)
    make_field(tmp_path)
    try:
        label_tool.save_label_map("field_9", "2025-06", tmp_path / "x.html")
    except ValueError:
        return
    raise AssertionError("июнь не должен приниматься")
