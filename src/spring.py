"""
spring.py: «был ли пиксель затоплен весной» (issue #19).

Рис заливают в разное время (с конца апреля до середины июня), поэтому одного снимка мало.
Мы просим у Copernicus ОДИН файл на поле, в котором сервер уже сам посчитал:
  канал 1: максимум NDWI по всем безоблачным снимкам с 20 апреля по 20 июня;
  канал 2: сколько снимков реально увидели этот пиксель (без облаков и теней).
Облака и тени убираем по каналу SCL (классификация сцены): иначе тень или облако
дали бы ложную «воду», а максимум по многим датам такие выбросы только усиливает.

Файл кладётся в кэш и повторно не скачивается.
"""
import numpy as np
import rasterio
import requests

import download
import rice

# Классы SCL, которые считаем плохими: 0 нет данных, 1 пересвет, 3 тень облака,
# 8 и 9 облака, 10 тонкие перистые облака, 11 снег.
BAD_SCL = [0, 1, 3, 8, 9, 10, 11]

# Evalscript: короткая программа, которая выполняется на сервере для каждого пикселя.
# mosaicking ORBIT: программа получает все снимки (пролёты спутника) за период сразу.
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B08", "SCL"], units: ["REFLECTANCE", "REFLECTANCE", "DN"] }],
    output: { bands: 2, sampleType: "FLOAT32" },
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(samples) {
  var bad = %s;
  var best = -2;   // -2: ни одного хорошего снимка
  var count = 0;
  for (var i = 0; i < samples.length; i++) {
    var s = samples[i];
    if (bad.indexOf(s.SCL) >= 0 || s.B03 + s.B08 <= 0) continue;
    var ndwi = (s.B03 - s.B08) / (s.B03 + s.B08);
    if (ndwi > best) best = ndwi;
    count++;
  }
  return [best, count];
}
""" % BAD_SCL


def build_request(geometry):
    """Тело запроса к Process API: один файл «максимум NDWI за весну» для контура."""
    bbox, epsg, width, height = download.field_bbox(geometry)
    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/%d" % epsg},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": rice.SPRING_FROM + "T00:00:00Z",
                                  "to": rice.SPRING_TO + "T23:59:59Z"},
                    "maxCloudCoverage": download.MAX_CLOUDS,
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


def fetch(geometry, tif, get_token):
    """
    Кладёт весенний файл в tif. Если файл уже есть в кэше, ничего не скачивает.
    get_token: функция, которая возвращает токен (вызывается, только если нужна сеть).
    """
    if tif.exists():
        return
    if not download.download_tif(get_token(), build_request(geometry), tif):
        raise requests.RequestException("сервер Copernicus вернул ошибку")


def read(tif, shape):
    """
    Читает весенний файл. Возвращает (flood, images):
      flood  - максимум NDWI за весну, NaN там, где не было ни одного хорошего снимка
      images - сколько весенних снимков видели хотя бы часть поля (0, если снимков нет)
    shape - форма снимка поля (высота, ширина): весенний файл должен лежать на той же сетке.
    """
    with rasterio.open(tif) as src:
        data = src.read().astype("float32")
    if data.shape[1:] != tuple(shape):
        raise ValueError("весенний снимок %s другого размера, удалите его из кэша" % tif.name)
    count = data[1]
    flood = np.where(count > 0, data[0], np.nan).astype("float32")
    return flood, int(count.max())


def flood(geometry, tif, shape, get_token):
    """
    Всё вместе: скачать (или взять из кэша) и прочитать. Возвращает (flood, images).
    Если весенних безоблачных снимков нет, возвращает (None, 0): маска риса тогда
    строится по летнему NDVI, а в ответе появляется пометка «без проверки затопления».
    """
    fetch(geometry, tif, get_token)
    flood_ndwi, images = read(tif, shape)
    if images == 0:
        return None, 0
    return flood_ndwi, images
