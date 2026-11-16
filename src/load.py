"""
load.py: чтение снимка поля и обрезка по границе поля.

Снимок data/<поле>_<месяц>.tif покрывает прямоугольник вокруг поля,
а само поле (контур из fields/*.geojson) может быть неправильной формы.
Чтобы считать только пиксели внутри поля, строим маску: True = пиксель внутри поля.

Главная функция:
    scene = load_scene("field_1", 7)
"""
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

import download  # берём из download.py пути, список каналов и чтение GeoJSON


def load_scene(field, month):
    """
    Читает снимок поля за месяц. Возвращает словарь:
      bands    {"B02": массив, "B03": ..., ...}  отражение 0..1 (float32)
      mask     True там, где пиксель внутри поля и в нём есть данные
      transform, crs   привязка к земле (нужны для карты)
      pixel_ha площадь одного пикселя в гектарах (10 x 10 м = 0.01 га)
    """
    tif = download.tif_path(field, month)
    with rasterio.open(tif) as src:
        data = src.read().astype("float32")   # форма (каналы, высота, ширина)
        transform = src.transform
        crs = src.crs

    # каналы по именам: порядок в файле такой же, как в download.BANDS
    bands = {name: data[i] for i, name in enumerate(download.BANDS)}

    # контур поля переводим из градусов в систему координат снимка (метры)
    geometry = download.read_geometry(download.FIELDS_DIR / (field + ".geojson"))
    geometry = transform_geom("EPSG:4326", crs, geometry)

    # invert=True: внутри контура True
    inside = geometry_mask([geometry], out_shape=data.shape[1:], transform=transform, invert=True)
    # в пикселях без данных (спутник там не снимал) все каналы равны 0
    has_data = data.any(axis=0)

    # площадь пикселя: ширина * высота в метрах, 1 га = 10 000 м²
    pixel_ha = abs(transform.a * transform.e) / 10000

    return {
        "bands": bands,
        "mask": inside & has_data,
        "transform": transform,
        "crs": crs,
        "pixel_ha": pixel_ha,
    }
