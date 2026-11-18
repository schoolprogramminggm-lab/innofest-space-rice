"""
spring.py: «был ли пиксель затоплен весной» (issue #19).

Рис заливают в разное время (с конца апреля до середины июня), поэтому одного снимка мало.
Мы просим у Copernicus ОДИН файл на поле, в котором сервер уже сам проверил каждый безоблачный
снимок с 20 апреля по 20 июня по правилу Xiao et al., 2005 (rice.py):
  канал 1: 1, если хотя бы на одном снимке LSWI + 0.05 >= NDVI (пиксель был затоплен), иначе 0;
  канал 2: сколько снимков реально увидели этот пиксель (без облаков и теней).
Облака и тени убираем по каналу SCL (классификация сцены): облако даёт NDVI около 0 и
LSWI около 0, то есть выглядело бы как «вода», а «хотя бы на одном снимке» такие выбросы усиливает.

Файл кладётся в кэш и повторно не скачивается.
"""
import rasterio
import requests

import download
import rice

# Классы SCL, которые считаем плохими: 0 нет данных, 1 пересвет, 3 тень облака,
# 8 и 9 облака, 10 тонкие перистые облака, 11 снег.
BAD_SCL = [0, 1, 3, 8, 9, 10, 11]

# Имя файла в кэше. Суффикс lswi: старые файлы «максимум NDWI» (прошлая версия правила) не подходят.
CACHE_NAME = "spring_lswi"

# Evalscript: короткая программа, которая выполняется на сервере для каждого пикселя.
# mosaicking ORBIT: программа получает все снимки (пролёты спутника) за период сразу.
# Формула затопления здесь та же, что rice.is_flooded.
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "B11", "SCL"],
              units: ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "DN"] }],
    output: { bands: 2, sampleType: "FLOAT32" },
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(samples) {
  var bad = %s;
  var flooded = 0;   // 1, если хотя бы на одном снимке LSWI + margin >= NDVI
  var count = 0;     // сколько хороших снимков видели этот пиксель
  for (var i = 0; i < samples.length; i++) {
    var s = samples[i];
    if (bad.indexOf(s.SCL) >= 0 || s.B08 + s.B04 <= 0 || s.B08 + s.B11 <= 0) continue;
    var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
    var lswi = (s.B08 - s.B11) / (s.B08 + s.B11);
    if (lswi + %s >= ndvi) flooded = 1;
    count++;
  }
  return [flooded, count];
}
""" % (BAD_SCL, rice.LSWI_MARGIN)


def build_request(geometry):
    """Тело запроса к Process API: один файл «был затоплен весной» для контура."""
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
    Читает весенний файл. Возвращает (flooded, images):
      flooded - True там, где пиксель был затоплен хотя бы на одном весеннем снимке
      images  - сколько весенних снимков видели хотя бы часть поля (0, если снимков нет)
    shape - форма снимка поля (высота, ширина): весенний файл должен лежать на той же сетке.
    """
    with rasterio.open(tif) as src:
        data = src.read().astype("float32")
    if data.shape[1:] != tuple(shape):
        raise ValueError("весенний снимок %s другого размера, удалите его из кэша" % tif.name)
    flooded = data[0] > 0.5
    return flooded, int(data[1].max())


def flood(geometry, tif, shape, get_token):
    """
    Всё вместе: скачать (или взять из кэша) и прочитать. Возвращает (flooded, images).
    Если весенних безоблачных снимков нет, возвращает (None, 0): рисовый массив тогда
    строится по летнему NDVI, а в ответе появляется пометка «без проверки затопления».
    """
    fetch(geometry, tif, get_token)
    flooded, images = read(tif, shape)
    if images == 0:
        return None, 0
    return flooded, images
