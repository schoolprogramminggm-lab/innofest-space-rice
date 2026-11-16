"""
download.py: скачивает снимки Sentinel-2 L2A для каждого поля.

Что делает:
  для каждого файла fields/*.geojson и для каждого месяца (май, июль, август 2025)
  просит у Copernicus Data Space самый безоблачный снимок месяца (облачность < 20%)
  и сохраняет:
    data/<поле>_<месяц>.tif    каналы B02, B03, B04, B08, B11
    output/<поле>_<месяц>.png  цветное превью (B04, B03, B02)

Ключи берутся из файла .env (CDSE_CLIENT_ID, CDSE_CLIENT_SECRET) и нигде не печатаются.

Запуск:  python src/download.py
"""
import json
import os
from pathlib import Path

import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from rasterio.warp import transform_bounds

# ---------- Настройки ----------
ROOT = Path(__file__).resolve().parent.parent   # корень проекта
FIELDS_DIR = ROOT / "fields"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

YEAR = 2025
MONTHS = [5, 7, 8]          # май, июль, август
MAX_CLOUDS = 20             # облачность меньше 20 %
PIXEL_SIZE = 10             # метров в одном пикселе
BANDS = ["B02", "B03", "B04", "B08", "B11"]

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Evalscript: короткая программа, которая говорит серверу, какие каналы вернуть.
# Результат: один файл с 5 каналами, числа с плавающей точкой (отражение от 0 до 1).
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: %s, units: "reflectance" }],
    output: { bands: %d, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [%s];
}
""" % (
    json.dumps(BANDS),
    len(BANDS),
    ", ".join("s." + b for b in BANDS),
)


# ---------- Вспомогательные функции (их проверяют тесты) ----------
def field_name(geojson_path):
    """fields/field_1.geojson -> 'field_1'"""
    return Path(geojson_path).stem


def month_label(month):
    """7 -> '2025-07'"""
    return "%d-%02d" % (YEAR, month)


def tif_path(field, month):
    """Куда сохранить снимок: data/field_1_2025-07.tif"""
    return DATA_DIR / ("%s_%s.tif" % (field, month_label(month)))


def png_path(field, month):
    """Куда сохранить превью: output/field_1_2025-07.png"""
    return OUTPUT_DIR / ("%s_%s.png" % (field, month_label(month)))


def month_range(month):
    """Первое и последнее время месяца в формате, который понимает Sentinel Hub."""
    # день 31 есть не во всех месяцах, поэтому считаем начало следующего месяца
    next_year, next_month = (YEAR + 1, 1) if month == 12 else (YEAR, month + 1)
    start = "%d-%02d-01T00:00:00Z" % (YEAR, month)
    end = "%d-%02d-01T00:00:00Z" % (next_year, next_month)
    return start, end


def read_geometry(geojson_path):
    """Достаёт контур поля из GeoJSON (Feature, FeatureCollection или просто геометрия)."""
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)
    if data["type"] == "FeatureCollection":
        data = data["features"][0]
    if data["type"] == "Feature":
        data = data["geometry"]
    return data


def utm_epsg(lon):
    """Номер системы координат UTM (в метрах) для северного полушария. Кызылорда: зона 41."""
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone


def field_bbox(geometry):
    """
    Рамка вокруг поля в метрах (UTM), края округлены до 10 м,
    чтобы пиксели лежали ровной сеткой. Возвращает (bbox, epsg, ширина, высота).
    """
    # рамка поля в градусах
    coords = geometry["coordinates"]
    flat = np.array(_flatten(coords))
    lon_min, lat_min = flat.min(axis=0)
    lon_max, lat_max = flat.max(axis=0)
    epsg = utm_epsg((lon_min + lon_max) / 2)

    # переводим в метры
    x1, y1, x2, y2 = transform_bounds("EPSG:4326", "EPSG:%d" % epsg,
                                      lon_min, lat_min, lon_max, lat_max)
    # округляем наружу до целого числа пикселей
    x1 = np.floor(x1 / PIXEL_SIZE) * PIXEL_SIZE
    y1 = np.floor(y1 / PIXEL_SIZE) * PIXEL_SIZE
    x2 = np.ceil(x2 / PIXEL_SIZE) * PIXEL_SIZE
    y2 = np.ceil(y2 / PIXEL_SIZE) * PIXEL_SIZE
    width = int((x2 - x1) / PIXEL_SIZE)
    height = int((y2 - y1) / PIXEL_SIZE)
    return [float(x1), float(y1), float(x2), float(y2)], epsg, width, height


def _flatten(coords):
    """Превращает вложенные списки координат в плоский список пар [lon, lat]."""
    if isinstance(coords[0], (int, float)):
        return [coords[:2]]
    result = []
    for part in coords:
        result.extend(_flatten(part))
    return result


def build_request(geometry, month):
    """Собирает тело запроса к Process API (без сети, поэтому легко тестировать)."""
    bbox, epsg, width, height = field_bbox(geometry)
    start, end = month_range(month)
    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/%d" % epsg},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": start, "to": end},
                    "maxCloudCoverage": MAX_CLOUDS,
                    # из всех снимков месяца берём самый безоблачный
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }


# ---------- Работа с сетью ----------
def get_token():
    """Получает временный токен по ключам из .env. Ключи не печатаем."""
    client_id = os.getenv("CDSE_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit("Нет ключей: заполните CDSE_CLIENT_ID и CDSE_CLIENT_SECRET в файле .env")
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }, timeout=60)
    if resp.status_code != 200:
        raise SystemExit("Не удалось войти в Copernicus (код %d). Проверьте ключи в .env." % resp.status_code)
    return resp.json()["access_token"]


def download_tif(token, request_body, out_path):
    """Отправляет запрос и сохраняет полученный GeoTIFF."""
    resp = requests.post(
        PROCESS_URL,
        headers={"Authorization": "Bearer " + token, "Accept": "image/tiff"},
        json=request_body,
        timeout=300,
    )
    if resp.status_code != 200:
        # печатаем только код и текст ответа сервера, заголовки с токеном не трогаем
        print("   ошибка сервера %d: %s" % (resp.status_code, resp.text[:200]))
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return True


def make_preview(tif, png):
    """Цветное превью из каналов B04 (красный), B03 (зелёный), B02 (синий)."""
    with rasterio.open(tif) as src:
        data = src.read().astype("float32")    # форма: (каналы, высота, ширина)
    # порядок каналов в файле как в BANDS: B02=0, B03=1, B04=2
    rgb = np.stack([data[2], data[1], data[0]], axis=-1)
    rgb = np.clip(rgb / 0.3, 0, 1)             # растягиваем яркость, чтобы снимок не был тёмным
    png.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.imsave(png, rgb)


def is_empty(tif):
    """True, если в снимке одни нули: за этот месяц нет снимка с облачностью < 20%."""
    with rasterio.open(tif) as src:
        return not src.read().any()


def main():
    load_dotenv(ROOT / ".env")

    fields = sorted(FIELDS_DIR.glob("*.geojson"))
    if not fields:
        raise SystemExit("В папке fields/ нет файлов *.geojson. Сначала нарисуйте границы полей (issue #3).")

    token = get_token()
    print("Вход в Copernicus выполнен.")

    for geojson in fields:
        field = field_name(geojson)
        geometry = read_geometry(geojson)
        for month in MONTHS:
            tif = tif_path(field, month)
            print("%s, %s" % (field, month_label(month)))
            if tif.exists():
                print("   уже скачано, пропускаю")
                continue
            body = build_request(geometry, month)
            if not download_tif(token, body, tif):
                continue
            if is_empty(tif):
                tif.unlink()
                print("   нет снимков с облачностью < %d%% за этот месяц" % MAX_CLOUDS)
                continue
            make_preview(tif, png_path(field, month))
            print("   сохранено: %s" % tif.name)


if __name__ == "__main__":
    main()
