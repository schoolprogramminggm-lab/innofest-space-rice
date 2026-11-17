"""
run.py: одна команда делает всё: индексы -> зоны -> карты.

Запуск:
    python run.py                   # все поля из fields/
    python run.py --field field_1   # только одно поле

    # любое место: квадрат 500 x 500 м вокруг точки (широта,долгота), снимок качается на лету
    python run.py --point 44.9716,64.5267 --month 2025-07
    python run.py --point 44.9716,64.5267 --size 800
    python run.py --geojson my_field.geojson --month 2025-08    # свой контур

Результат в папке output/:
    ndvi_<поле>_<месяц>.png   картинка NDVI
    map_<поле>.html           карта поля с зонами за все месяцы
    summary.json              площади зон в гектарах
    <поле>_<месяц>.json       ответ для бота AgroQyzyl (docs/agroqyzyl-format.md)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # чтобы работало `import download`, `import zones`...

import download
import export
import indices
import load
import maps
import point
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


def run_point(args):
    """Режим «любое место»: --point или --geojson. Печатает ответ и время работы."""
    start = time.time()
    sys.stdout.reconfigure(errors="replace")   # чтобы казахские буквы не роняли печать в консоли

    if args.point:
        try:
            lat, lon = point.parse_point(args.point)
        except ValueError as e:
            raise SystemExit("Ошибка в --point: %s" % e)
        answer = point.analyze_point(lat, lon, args.size, args.month)
    else:
        answer = point.analyze_geojson(args.geojson, args.month)

    if answer["ok"]:
        result = answer["result"]
        print("%s, %s (участок %.1f га)" % (result["field"], result["date"], result["area_ha"]))
        for key in zones.ZONE_KEYS:
            z = result["zones"][key]
            print("  %-14s %6.1f га  (%4.1f%%)" % (zones.ZONE_NAMES_RU[key], z["ha"], z["pct"]))
        # JSON для AgroQyzyl рядом с картой
        path = point.OUTPUT_DIR / ("%s_%s.json" % (result["field"], result["date"]))
        path.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("  карта: %s" % result["map_html"])
        print("  NDVI:  %s" % result["map_png"])
        print("  JSON для AgroQyzyl: %s" % path.relative_to(ROOT))
    else:
        print("Нет результата: %s" % answer["error"])
    print("\n" + answer["message"]["ru"])
    print(answer["message"]["kk"])

    seconds = time.time() - start
    print("\nЗаняло %.1f с%s" % (seconds, "" if seconds <= 60 else " (дольше 60 с!)"))


def main():
    parser = argparse.ArgumentParser(description="Индексы, зоны и карты полей")
    parser.add_argument("--field", help="имя поля, например field_1 (по умолчанию все)")
    parser.add_argument("--point", help="широта,долгота, например 44.9716,64.5267: любое место")
    parser.add_argument("--geojson", help="путь к GeoJSON со своим контуром поля")
    parser.add_argument("--size", type=int, default=500, help="сторона квадрата вокруг точки, м (по умолчанию 500)")
    parser.add_argument("--month", default=point.DEFAULT_MONTH, help="месяц сезона: 2025-05, 2025-07 или 2025-08")
    args = parser.parse_args()

    if args.point or args.geojson:
        if args.point and args.geojson:
            raise SystemExit("Укажите что-то одно: --point или --geojson")
        run_point(args)
        return

    geojsons = sorted(download.FIELDS_DIR.glob("*.geojson"))
    if args.field:
        geojsons = [g for g in geojsons if download.field_name(g) == args.field]
    if not geojsons:
        raise SystemExit("Не найдены поля в fields/")

    # Если summary.json уже есть, дополняем его: так запуск с --field не сотрёт остальные поля
    summary = export.load_summary() or {}
    done = []  # какие (поле, месяц) посчитаны сейчас: для них напишем JSON для AgroQyzyl
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
            done.append((field, label))

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

    # JSON для AgroQyzyl: по файлу на каждое поле и месяц (формат: docs/agroqyzyl-format.md)
    for field, label in done:
        path = export.save_agroqyzyl(field, label, summary)
        print("  JSON для AgroQyzyl: %s" % path.relative_to(ROOT))


if __name__ == "__main__":
    main()
