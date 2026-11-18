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
import zones

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
    assert [k for k, _, _ in label_tool.CLASSES] == zones.POINT_CLASSES   # классы разметки = классы модели
    assert "Скачать CSV" in html and "points_' + FIELD + '.csv" in html and "labels/points_field_9.csv" in html
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


# ---------- Уже сохранённые точки загружаются на карту (issue #10) ----------
def write_points(folder, field, rows):
    """labels/points_<поле>.csv: rows = [(lat, lon, класс, дата), ...]"""
    lines = [label_tool.CSV_HEADER] + ["%s,%s,%s,%s,%s" % (lat, lon, cls, field, d) for lat, lon, cls, d in rows]
    (folder / ("points_%s.csv" % field)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_points_skips_bad_rows(tmp_path):
    write_points(tmp_path, "field_9", [(44.98, 64.52, "healthy", "2025-07"), (44.99, 64.53, "salinity", "2025-05")])
    with open(tmp_path / "points_field_9.csv", "a", encoding="utf-8") as f:
        f.write("44.97,64.51,dragon,field_9,2025-07\nabc,64.5,healthy,field_9,2025-07\n")   #неизвестный класс, не число
    points, skipped = label_tool.read_points(tmp_path / "points_field_9.csv")
    assert [p["class"] for p in points] == ["healthy", "salinity"] and skipped == 2
    assert points[1]["date"] == "2025-05" and points[0]["lat"] == 44.98
    assert label_tool.read_points(tmp_path / "no_such_file.csv") == ([], 0)    # файла нет: это не ошибка


def test_saved_points_are_shown_on_the_map(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download, "FIELDS_DIR", tmp_path)
    monkeypatch.setattr(label_tool, "LABELS_DIR", tmp_path)
    make_field(tmp_path)
    write_points(tmp_path, "field_9", [(44.98, 64.52, "healthy", "2025-07"), (44.981, 64.521, "salinity", "2025-07")])

    html = label_tool.save_label_map("field_9", "2025-07", tmp_path / "label.html").read_text(encoding="utf-8")
    assert "var EXISTING = [{" in html and '"cls": "salinity"' in html and '"lat": 44.981' in html
    assert "Загружено из labels/points_field_9.csv: 2" in html

    # новая точка получит дату этого снимка, а у загруженной остаётся своя (поэтому CSV пишет p.date || DATE)
    assert "p.date || DATE" in html


def test_map_without_saved_file_starts_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download, "FIELDS_DIR", tmp_path)
    monkeypatch.setattr(label_tool, "LABELS_DIR", tmp_path)
    make_field(tmp_path)
    html = label_tool.save_label_map("field_9", "2025-07", tmp_path / "label.html").read_text(encoding="utf-8")
    assert "var EXISTING = [];" in html and "разметка с нуля" in html


def test_storage_key_changes_when_file_changes():
    """Положили новый CSV: старая работа в браузере не подмешивается к нему."""
    a = [{"lat": 1.0, "lon": 2.0, "class": "healthy", "field": "f", "date": "2025-07"}]
    b = a + [{"lat": 1.5, "lon": 2.5, "class": "water", "field": "f", "date": "2025-07"}]
    key = label_tool.storage_key("f", "2025-07", a)
    assert key == label_tool.storage_key("f", "2025-07", list(a))
    assert key != label_tool.storage_key("f", "2025-07", b)
