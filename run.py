"""
run.py: одна команда делает всё: индексы -> зоны -> карты.

Запуск:
    python run.py                   # все поля из fields/
    python run.py --field field_1   # только одно поле

Результат в папке output/:
    ndvi_<поле>_<месяц>.png   картинка NDVI
    map_<поле>.html           карта поля с зонами за все месяцы
    summary.json              площади зон в гектарах
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # чтобы работало `import download`, `import zones`...

import download
import indices
import load
import maps
import zones


def process_scene(field, month):
    """Один снимок: считаем индексы, рисуем NDVI, делим на зоны."""
    scene = load.load_scene(field, month)
    b = scene["bands"]

    # 1. индексы
    ndvi = indices.ndvi(b["B04"], b["B08"])
    si = indices.si(b["B02"], b["B04"])
    ndwi = indices.ndwi(b["B03"], b["B08"])

    # 2. картинка NDVI
    label = "%s, %s" % (field, download.month_label(month))
    indices.save_ndvi_png(
        ndvi, scene["mask"], "NDVI: " + label,
        download.OUTPUT_DIR / ("ndvi_%s_%s.png" % (field, download.month_label(month))),
    )

    # 3. зоны KMeans
    zone_map, means = zones.find_zones(ndvi, si, ndwi, scene["mask"])
    areas = zones.zone_areas(zone_map, scene["pixel_ha"])
    field_ha = round(float(scene["mask"].sum()) * scene["pixel_ha"], 1)
    return zone_map, scene, areas, field_ha, means


def main():
    parser = argparse.ArgumentParser(description="Индексы, зоны и карты полей")
    parser.add_argument("--field", help="имя поля, например field_1 (по умолчанию все)")
    args = parser.parse_args()

    geojsons = sorted(download.FIELDS_DIR.glob("*.geojson"))
    if args.field:
        geojsons = [g for g in geojsons if download.field_name(g) == args.field]
    if not geojsons:
        raise SystemExit("Не найдены поля в fields/")

    summary = {}
    for geojson in geojsons:
        field = download.field_name(geojson)
        geometry = download.read_geometry(geojson)
        summary[field] = {}
        map_scenes = {}  # для карты: {месяц: (зоны, transform, crs)}

        for month in download.MONTHS:
            label = download.month_label(month)
            if not download.tif_path(field, month).exists():
                print("%s %s: нет снимка в data/, пропускаю" % (field, label))
                continue

            zone_map, scene, areas, field_ha, means = process_scene(field, month)
            map_scenes[month] = (zone_map, scene["transform"], scene["crs"])
            summary[field][label] = {"area_ha": field_ha, "zones": areas}

            # площади зон в консоль
            print("\n%s, %s (поле %.1f га)" % (field, label, field_ha))
            for key in zones.ZONE_KEYS:
                print("  %-14s %6.1f га  (%4.1f%%)" % (
                    zones.ZONE_NAMES_RU[key], areas[key]["ha"], areas[key]["pct"]))

        if map_scenes:
            map_path = download.OUTPUT_DIR / ("map_%s.html" % field)
            maps.save_map(field, geometry, map_scenes, map_path)
            print("  карта: %s" % map_path.relative_to(ROOT))

    # площади всех зон в JSON
    out = download.OUTPUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nГотово. Площади: %s" % out.relative_to(ROOT))


if __name__ == "__main__":
    main()
