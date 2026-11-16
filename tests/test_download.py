"""Тесты download.py без интернета: проверяем запрос и имена файлов."""
import json
import sys
from pathlib import Path

# чтобы можно было сделать import download
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import download

# маленький квадрат около Кызылорды (примерно 1 км x 1 км)
FIELD = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[65.50, 44.80], [65.51, 44.80], [65.51, 44.81], [65.50, 44.81], [65.50, 44.80]]],
    },
}


def test_file_names():
    assert download.field_name("fields/field_1.geojson") == "field_1"
    assert download.tif_path("field_1", 7).name == "field_1_2025-07.tif"
    assert download.tif_path("field_2", 5).parent.name == "data"
    assert download.png_path("field_3", 8).name == "field_3_2025-08.png"
    assert download.png_path("field_3", 8).parent.name == "output"


def test_month_range():
    assert download.month_range(5) == ("2025-05-01T00:00:00Z", "2025-06-01T00:00:00Z")
    assert download.month_range(8) == ("2025-08-01T00:00:00Z", "2025-09-01T00:00:00Z")


def test_request():
    body = download.build_request(FIELD["geometry"], 7)
    data_filter = body["input"]["data"][0]["dataFilter"]

    assert body["input"]["data"][0]["type"] == "sentinel-2-l2a"
    assert data_filter["maxCloudCoverage"] == 20
    assert data_filter["timeRange"]["from"] == "2025-07-01T00:00:00Z"
    assert data_filter["timeRange"]["to"] == "2025-08-01T00:00:00Z"
    assert data_filter["mosaickingOrder"] == "leastCC"

    # все пять каналов запрошены
    for band in ["B02", "B03", "B04", "B08", "B11"]:
        assert band in body["evalscript"]

    # размер картинки совпадает с рамкой: 1 пиксель = 10 м
    x1, y1, x2, y2 = body["input"]["bounds"]["bbox"]
    assert body["output"]["width"] == round((x2 - x1) / 10)
    assert body["output"]["height"] == round((y2 - y1) / 10)
    # квадрат ~1 км -> около 100 пикселей
    assert 80 < body["output"]["width"] < 120
    assert 80 < body["output"]["height"] < 140

    # Кызылорда: UTM зона 41
    assert body["input"]["bounds"]["properties"]["crs"].endswith("/32641")
    # запрос должен превращаться в JSON
    json.dumps(body)


def test_read_geometry(tmp_path):
    path = tmp_path / "field_1.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [FIELD]}), encoding="utf-8")
    assert download.read_geometry(path)["type"] == "Polygon"
