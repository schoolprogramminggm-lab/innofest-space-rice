"""
maps.py: интерактивная HTML-карта поля (folium).

На карте: спутниковая подложка, контур поля, цветные зоны за май, июль и август
(месяцы переключаются в панели справа) и легенда на казахском и русском.
Карте для показа нужен интернет (подложка грузится из сети), см. README, раздел 10.
"""
from pathlib import Path

import folium
import numpy as np
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

import zones

# Цвета зон (RGB), как в README: зелёный, жёлтый, белый, синий, коричневый, серый. Порядок как в zones.ZONE_KEYS.
ZONE_COLORS = [
    (46, 158, 62),    # healthy  — зелёный
    (242, 197, 0),    # stress   — жёлтый
    (245, 245, 245),  # salinity — белый (соль)
    (43, 108, 222),   # water    — синий
    (181, 136, 99),   # empty    — коричневый (неудачный посев / пусто)
    (128, 128, 128),  # not_rice — серый (не рисовое поле)
]

# Подписи зон для легенды: (русский, казахский)
ZONE_LABELS = [
    ("Здоровый рис", "Сау күріш"),
    ("Стресс", "Күйзеліс"),
    ("Засоление", "Тұзданған жер"),
    ("Вода", "Су"),
    ("Неудачный посев / пусто", "Сәтсіз егіс / бос жер"),
    ("Не рисовое поле", "Күріш алқабы емес"),
]

# Названия месяцев для переключателя слоёв: (русский, казахский)
MONTH_LABELS = {
    5: ("Май", "Мамыр"),
    7: ("Июль", "Шілде"),
    8: ("Август", "Тамыз"),
}

# Спутниковая подложка Esri World Imagery (бесплатная)
SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def zones_to_latlon(zone_map, transform, crs):
    """
    Переводит карту зон из метров (UTM) в широту/долготу.
    Нужно, чтобы зоны на карте точно совпали с контуром поля: сетка UTM слегка
    повёрнута относительно параллелей и меридианов.
    Возвращает (карта зон в градусах, границы [[юг, запад], [север, восток]]).
    """
    height, width = zone_map.shape
    left, bottom, right, top = array_bounds(height, width, transform)
    new_transform, new_width, new_height = calculate_default_transform(
        crs, "EPSG:4326", width, height, left, bottom, right, top
    )

    result = np.zeros((new_height, new_width), dtype="uint8")
    reproject(
        source=zone_map, destination=result,
        src_transform=transform, src_crs=crs,
        dst_transform=new_transform, dst_crs="EPSG:4326",
        src_nodata=0, dst_nodata=0,
        resampling=Resampling.nearest,  # nearest: коды зон не смешиваются
    )

    west, south, east, north = array_bounds(new_height, new_width, new_transform)
    return result, [[south, west], [north, east]]


def zones_to_rgba(zone_map):
    """Раскрашивает карту зон: RGBA-картинка, вне поля прозрачно."""
    rgba = np.zeros(zone_map.shape + (4,), dtype="uint8")
    for i, color in enumerate(ZONE_COLORS):
        rgba[zone_map == i + 1] = (*color, 255)
    return rgba


def legend_html():
    """Легенда: цветной квадрат и название зоны на двух языках (русский / казахский)."""
    rows = ""
    for (ru, kk), (r, g, b) in zip(ZONE_LABELS, ZONE_COLORS):
        rows += (
            '<div style="margin:3px 0;">'
            '<span style="display:inline-block;width:16px;height:16px;'
            'background:rgb(%d,%d,%d);border:1px solid #555;margin-right:8px;'
            'vertical-align:middle;"></span>%s / %s</div>' % (r, g, b, ru, kk)
        )
    return (
        '<div style="position:fixed;bottom:30px;left:10px;z-index:9999;'
        'background:white;padding:10px 14px;border:2px solid #888;border-radius:6px;'
        'font-size:14px;font-family:sans-serif;">'
        '<b>Зоны поля / Алқап аймақтары</b>' + rows + '</div>'
    )


def point_marker(lat, lon, point_class):
    """Маркер точки запроса: цвет и подпись по её классу (zones.POINT_CLASSES)."""
    index = zones.ZONE_KEYS.index(point_class)
    r, g, b = ZONE_COLORS[index]
    ru, kk = ZONE_LABELS[index]
    return folium.CircleMarker(
        [lat, lon], radius=8, color="#000000", weight=2, fill=True,
        fill_color="#%02x%02x%02x" % (r, g, b), fill_opacity=1,
        tooltip="Точка: %s / %s (%s)" % (ru, kk, point_class),
    )


def build_map(field, geometry, scenes, year=2025, marker=None):
    """
    Собирает карту поля.

    field    — имя поля, например "field_1"
    geometry — контур поля (GeoJSON, градусы)
    scenes   — {месяц: (карта зон, transform, crs)}, например {5: (...), 7: (...), 8: (...)}
    marker   — (широта, долгота, класс) точки запроса или None: на карте появится маркер
    Возвращает объект folium.Map.
    """
    # переводим зоны каждого месяца в градусы и находим границы поля
    layers = {}
    for month, (zone_map, transform, crs) in scenes.items():
        layers[month] = zones_to_latlon(zone_map, transform, crs)

    any_bounds = next(iter(layers.values()))[1]
    center = [(any_bounds[0][0] + any_bounds[1][0]) / 2, (any_bounds[0][1] + any_bounds[1][1]) / 2]

    m = folium.Map(location=center, zoom_start=15, tiles=None)
    folium.TileLayer(tiles=SATELLITE_TILES, attr="Esri World Imagery", name="Спутник").add_to(m)

    # зоны каждого месяца отдельным слоем; по умолчанию виден июль (или первый месяц)
    default_month = 7 if 7 in layers else next(iter(layers))
    for month in sorted(layers):
        zones, bounds = layers[month]
        ru, kk = MONTH_LABELS.get(month, (str(month), str(month)))
        folium.raster_layers.ImageOverlay(
            image=zones_to_rgba(zones),
            bounds=bounds,                 # [[юго-запад: широта, долгота], [северо-восток]]
            opacity=0.7,
            name="%s / %s %d" % (ru, kk, year),
            show=(month == default_month),
        ).add_to(m)

    # контур поля поверх зон: по нему видно, что зоны лежат точно на поле
    folium.GeoJson(
        geometry,
        name="Контур поля / Алқап шекарасы",
        style_function=lambda _: {"color": "#FF2D55", "weight": 3, "fillOpacity": 0},
        tooltip=field,
    ).add_to(m)

    if marker:
        point_marker(*marker).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html()))
    return m


def save_map(field, geometry, scenes, path, marker=None):
    """Строит карту и сохраняет в HTML-файл. marker: (широта, долгота, класс) или None."""
    m = build_map(field, geometry, scenes, marker=marker)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
