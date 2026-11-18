"""
spring.py: «был ли пиксель затоплен весной» (issue #19).

Рис заливают в разное время (с конца апреля до середины июня), поэтому одного снимка мало.
Мы просим у Copernicus ОДИН файл на поле. В нём по одному каналу на каждый пролёт спутника
с 20 апреля по 20 июня (облачность всей сцены не выше rice.SCENE_CLOUD_PREFILTER, это грубый
предфильтр). Значение пикселя в канале пролёта:
  0  снимок хороший, пиксель не затоплен
  1  снимок хороший, пиксель затоплен: LSWI + 0.05 >= NDVI (Xiao et al., 2005, см. rice.py)
  2  облако, тень, нет данных (по каналу SCL)
  3  такого пролёта нет (канал не использован)

Решать, какие снимки брать, мы будем уже в Python, по облачности НАД ПОЛЕМ: если в контуре
больше rice.MAX_FIELD_CLOUD пикселей со значением 2, снимок пропускаем. Поэтому порог можно
менять, не скачивая файл заново. Файл лежит в кэше и повторно не скачивается.
"""
import numpy as np
import rasterio
import requests

import download
import rice

# Классы SCL, которые считаем плохими: 0 нет данных, 1 пересвет, 3 тень облака,
# 8 и 9 облака, 10 тонкие перистые облака, 11 снег.
BAD_SCL = [0, 1, 3, 8, 9, 10, 11]

# Коды значений в файле
DRY, FLOODED, BAD, NO_ORBIT = 0, 1, 2, 3

# Сколько пролётов помещается в файл (по каналу на пролёт). За два месяца их около 15.
MAX_ORBITS = 40

# Имя файла в кэше. Старые файлы («максимум NDWI», «затоплен хотя бы раз») устроены иначе.
CACHE_NAME = "spring_orbits"

# Evalscript: короткая программа, которая выполняется на сервере для каждого пикселя.
# mosaicking ORBIT: программа получает все пролёты за период сразу (samples).
# Формула затопления здесь та же, что rice.is_flooded.
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "B11", "SCL"],
              units: ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "DN"] }],
    output: { bands: %(max)d, sampleType: "UINT8" },
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(samples) {
  var bad = %(bad)s;
  var out = [];
  for (var i = 0; i < %(max)d; i++) {
    if (i >= samples.length) { out.push(%(none)d); continue; }
    var s = samples[i];
    if (bad.indexOf(s.SCL) >= 0 || s.B08 + s.B04 <= 0 || s.B08 + s.B11 <= 0) { out.push(%(badcode)d); continue; }
    var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
    var lswi = (s.B08 - s.B11) / (s.B08 + s.B11);
    out.push(lswi + %(margin)s >= ndvi ? %(wet)d : %(dry)d);
  }
  return out;
}
""" % {"max": MAX_ORBITS, "bad": BAD_SCL, "none": NO_ORBIT, "badcode": BAD,
       "margin": rice.LSWI_MARGIN, "wet": FLOODED, "dry": DRY}


def build_request(geometry):
    """Тело запроса к Process API: один файл «пролёты за весну» для контура."""
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
                    "maxCloudCoverage": rice.SCENE_CLOUD_PREFILTER,
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
    Если сервер недоступен, поднимает download.CopernicusUnavailable.
    """
    if tif.exists():
        return
    if not download.download_tif(get_token(), build_request(geometry), tif):
        raise requests.RequestException("сервер Copernicus отклонил запрос")


def orbit_stats(tif, shape, mask):
    """
    Статистика по каждому пролёту: список словарей
      {"cloud": доля облаков и теней над полем (0..1), "flooded": доля затопленных пикселей поля,
       "used": True, если снимок берём (облаков над полем меньше rice.MAX_FIELD_CLOUD)}.
    Пролёты, которых нет (код 3 во всём файле), в список не попадают.
    mask - True внутри поля (облачность считаем только по нему).
    """
    with rasterio.open(tif) as src:
        data = src.read()
    if data.shape[1:] != tuple(shape):
        raise ValueError("весенний снимок %s другого размера, удалите его из кэша" % tif.name)
    stats = []
    total = int(mask.sum())
    for band in data:
        if (band == NO_ORBIT).all():
            continue
        cloud = float((band[mask] == BAD).sum()) / total if total else 1.0
        stats.append({
            "cloud": cloud,
            "flooded": float((band[mask] == FLOODED).sum()) / total if total else 0.0,
            "used": cloud < rice.MAX_FIELD_CLOUD,
        })
    return stats


def read(tif, shape, mask):
    """
    Читает весенний файл. Возвращает (flooded, images):
      flooded - True там, где пиксель был затоплен хотя бы на одном ПОДХОДЯЩЕМ снимке
      images  - сколько снимков подошло (над полем меньше rice.MAX_FIELD_CLOUD облаков и теней)
    Пиксель в облаке или тени на подходящем снимке просто не голосует за «затоплен».
    """
    with rasterio.open(tif) as src:
        data = src.read()
    if data.shape[1:] != tuple(shape):
        raise ValueError("весенний снимок %s другого размера, удалите его из кэша" % tif.name)
    total = int(mask.sum())
    flooded = np.zeros(tuple(shape), dtype=bool)
    images = 0
    for band in data:
        if (band == NO_ORBIT).all() or total == 0:
            continue
        cloud = float((band[mask] == BAD).sum()) / total
        if cloud >= rice.MAX_FIELD_CLOUD:
            continue                                  # слишком облачно над полем: снимок пропускаем
        images += 1
        flooded |= (band == FLOODED)
    return flooded, images


def flood(geometry, tif, shape, mask, get_token):
    """
    Всё вместе: скачать (или взять из кэша) и прочитать. Возвращает (flooded, images).
    Если подходящих весенних снимков нет, возвращает (None, 0): рисовый массив тогда
    строится по летнему NDVI, а в ответе появляется пометка «без проверки затопления».
    Ошибка сети сюда не относится: она поднимается как download.CopernicusUnavailable.
    """
    fetch(geometry, tif, get_token)
    flooded, images = read(tif, shape, mask)
    if images == 0:
        return None, 0
    return flooded, images
