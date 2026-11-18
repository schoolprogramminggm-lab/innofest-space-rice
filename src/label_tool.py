"""
label_tool.py: HTML-карта для ручной разметки точек (issue #10).

Запуск:
    python src/label_tool.py field_1                  # снимок июля (по умолчанию)
    python src/label_tool.py field_2 --month 2025-05

Результат: output/label_<поле>.html. Откройте файл в браузере (нужен интернет для подложки):
  1. справа выберите класс (здоровый рис / стресс / соль / вода / не рис);
  2. кликните по карте: появится точка цвета этого класса. Клик по точке удаляет её;
  3. кнопка «Скачать CSV» сохраняет файл points_<поле>.csv. Положите его в labels/.

Если файл labels/points_<поле>.csv уже есть, его точки загружаются на карту при запуске:
их можно дорисовывать и удалять, а не начинать заново. Скачанный CSV содержит старые
и новые точки вместе, поэтому просто замените им файл в labels/.

Строки CSV: lat, lon, class, field, date. Классы совпадают с zones.POINT_CLASSES
(healthy, stress, salinity, water, not_rice). date это месяц снимка, на котором ставили точку.
На карте два слоя: настоящий снимок Sentinel-2 (тот, по которому считает программа)
и спутниковая подложка Esri; переключаются в панели слоёв.

Точки хранятся в браузере (localStorage), поэтому обновление страницы их не стирает.
Новых библиотек не нужно: folium и немного JavaScript.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import folium
import numpy as np

import download
import load
import maps
import zones

# Классы разметки: (ключ в CSV, подпись, цвет точки). Ключи как в zones.ZONE_KEYS.
CLASSES = [
    ("healthy", "Здоровый рис", "#2e9e3e"),
    ("stress", "Стресс", "#f2c500"),
    ("salinity", "Соль", "#ffffff"),
    ("water", "Вода", "#2b6cde"),
    ("not_rice", "Не рис", "#808080"),
]

CSV_HEADER = "lat,lon,class,field,date"
LABELS_DIR = download.ROOT / "labels"   # сюда кладём points_<поле>.csv (тесты подменяют папку)


def points_path(field):
    """labels/points_<поле>.csv: файл с точками одного поля."""
    return LABELS_DIR / ("points_%s.csv" % field)


def read_points(path):
    """
    Читает файл с точками. Возвращает (точки, пропущено строк).
    Точка: {"lat", "lon", "class", "field", "date"}. Строка с неизвестным классом
    или нечисловыми координатами пропускается и считается в «пропущено».
    Если файла нет, возвращает ([], 0).
    """
    path = Path(path)
    if not path.exists():
        return [], 0
    points, skipped = [], 0
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                point = {"lat": float(row["lat"]), "lon": float(row["lon"]), "class": row["class"].strip(),
                         "field": (row.get("field") or "").strip(), "date": (row.get("date") or "").strip()}
            except (KeyError, TypeError, ValueError, AttributeError):
                skipped += 1
                continue
            if point["class"] not in zones.POINT_CLASSES:
                skipped += 1
                continue
            points.append(point)
    return points, skipped


def true_color_rgba(scene):
    """
    Настоящие цвета снимка (красный B04, зелёный B03, синий B02) как RGBA-картинка в градусах.
    Нужна, чтобы метить точки на том самом снимке, который видит программа.
    Возвращает (картинка, границы [[юг, запад], [север, восток]]).
    """
    b = scene["bands"]
    channels = []
    for name in ("B04", "B03", "B02"):
        # растягиваем яркость (0.3 = очень светло), иначе снимок тёмный; как в download.make_preview
        gray = (np.clip(b[name] / 0.3, 0, 1) * 255).astype("uint8")
        # maps.zones_to_latlon переводит массив из метров в градусы (nearest: яркость не смешивается)
        warped, bounds = maps.zones_to_latlon(gray, scene["transform"], scene["crs"])
        channels.append(warped)
    rgba = np.stack(channels + [np.where(channels[0] | channels[1] | channels[2], 255, 0).astype("uint8")], axis=-1)
    return rgba, bounds


def panel_html(field, month, loaded_note=""):
    """Панель справа: выбор класса, счётчик, кнопки. loaded_note: сколько точек загружено из файла."""
    rows = ""
    for i, (key, label, color) in enumerate(CLASSES):
        rows += (
            '<label style="display:block;margin:4px 0;cursor:pointer;">'
            '<input type="radio" name="cls" value="%s"%s> '
            '<span style="display:inline-block;width:14px;height:14px;background:%s;'
            'border:1px solid #555;vertical-align:middle;"></span> %d. %s</label>'
            % (key, " checked" if i == 0 else "", color, i + 1, label)
        )
    return (
        '<div style="position:fixed;top:10px;right:10px;z-index:9999;background:white;'
        'padding:10px 14px;border:2px solid #888;border-radius:6px;font:14px sans-serif;width:230px;">'
        '<b>Разметка: %s, %s</b>'
        '<div style="margin:6px 0;color:#555;">Выберите класс и кликните по карте. '
        'Клик по точке удаляет её.</div>%s'
        '<div style="margin:8px 0;">Точек: <b id="count">0</b></div>'
        '<div id="by_class" style="font-size:12px;color:#555;margin-bottom:8px;"></div>'
        '<button id="undo" style="margin-right:4px;">Отменить последнюю</button>'
        '<button id="download" style="margin-top:6px;font-weight:bold;">Скачать CSV</button>'
        '<div style="margin-top:6px;font-size:12px;color:#555;">Файл положите в labels/points_%s.csv</div>'
        '<div style="margin-top:4px;font-size:12px;color:#555;">%s</div>'
        '</div>' % (field, month, rows, field, loaded_note)
    )


def storage_key(field, month, existing):
    """
    Ключ localStorage. В него входит отпечаток точек из файла: если файл в labels/ изменился
    (например, вы положили новый CSV), старая работа в браузере не подмешивается к нему.
    """
    digest = hashlib.md5(json.dumps(existing, sort_keys=True).encode()).hexdigest()[:8]
    return "labels_%s_%s_%s" % (field, month, digest)


def script_js(map_name, field, month, existing=()):
    """
    JavaScript: клики, точки, счётчик, сохранение в браузере, скачивание CSV.
    existing: точки из labels/points_<поле>.csv (read_points), они показываются сразу.
    """
    classes = json.dumps([[k, c] for k, _, c in CLASSES])
    # Скрипт стоит в файле раньше, чем folium создаёт карту, поэтому ждём загрузки страницы
    return """
window.addEventListener('load', function () {
  var map = %(map)s;
  var FIELD = %(field)s, DATE = %(date)s, STORAGE = %(storage)s;
  var EXISTING = %(existing)s;   // точки из файла labels/points_<поле>.csv
  var COLORS = {};
  %(classes)s.forEach(function (c) { COLORS[c[0]] = c[1]; });
  var points = [];   // каждая точка: {lat, lon, cls, date, marker}

  function currentClass() {
    return document.querySelector('input[name="cls"]:checked').value;
  }

  function save() {   // запоминаем точки в браузере (если он это запрещает, просто пропускаем)
    try {
      localStorage.setItem(STORAGE, JSON.stringify(points.map(function (p) {
        return {lat: p.lat, lon: p.lon, cls: p.cls, date: p.date};
      })));
    } catch (e) {}
  }

  function refresh() {   // счётчик и сколько точек в каждом классе
    document.getElementById('count').textContent = points.length;
    var counts = {};
    points.forEach(function (p) { counts[p.cls] = (counts[p.cls] || 0) + 1; });
    document.getElementById('by_class').textContent = Object.keys(COLORS).map(function (k) {
      return k + ': ' + (counts[k] || 0);
    }).join(', ');
    save();
  }

  function addPoint(lat, lon, cls, date) {
    var marker = L.circleMarker([lat, lon], {
      radius: 6, color: '#000', weight: 1.5, fillColor: COLORS[cls], fillOpacity: 1,
      bubblingMouseEvents: false   // клик по точке не должен ставить новую точку под ней
    }).addTo(map);
    var point = {lat: lat, lon: lon, cls: cls, date: date, marker: marker};
    marker.on('click', function () {   // клик по точке удаляет её
      map.removeLayer(marker);
      points.splice(points.indexOf(point), 1);
      refresh();
    });
    points.push(point);
    refresh();
  }

  map.on('click', function (e) { addPoint(e.latlng.lat, e.latlng.lng, currentClass(), DATE); });

  document.getElementById('undo').onclick = function () {
    var last = points.pop();
    if (last) { map.removeLayer(last.marker); refresh(); }
  };

  document.getElementById('download').onclick = function () {
    var lines = ['%(header)s'];
    points.forEach(function (p) {
      // у старых точек остаётся их дата, у новых дата этого снимка
      lines.push([p.lat.toFixed(6), p.lon.toFixed(6), p.cls, FIELD, p.date || DATE].join(','));
    });
    var blob = new Blob([lines.join('\\n') + '\\n'], {type: 'text/csv;charset=utf-8'});
    var link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'points_' + FIELD + '.csv';
    link.click();
  };

  // Что показать при открытии: если в браузере есть сохранённая работа (она уже включает
  // точки из файла и правки), берём её; иначе точки из файла labels/points_<поле>.csv.
  var start = EXISTING.map(function (p) { return {lat: p.lat, lon: p.lon, cls: p.cls, date: p.date}; });
  try {
    var saved = localStorage.getItem(STORAGE);
    if (saved !== null) { start = JSON.parse(saved); }
  } catch (e) {}
  start.forEach(function (p) { addPoint(p.lat, p.lon, p.cls, p.date); });
  refresh();
});
""" % {
        "map": map_name,
        "field": json.dumps(field),
        "date": json.dumps(month),
        "storage": json.dumps(storage_key(field, month, existing)),
        "existing": json.dumps([{"lat": p["lat"], "lon": p["lon"], "cls": p["class"], "date": p["date"]}
                                for p in existing]),
        "classes": classes,
        "header": CSV_HEADER,
    }


def build_label_map(field, geometry, scene, month, existing=(), skipped=0):
    """
    Собирает карту для разметки. Возвращает объект folium.Map.
    existing - уже сохранённые точки поля (read_points), skipped - сколько строк файла не прочитано.
    """
    rgba, bounds = true_color_rgba(scene)
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]

    m = folium.Map(location=center, zoom_start=16, tiles=None, max_zoom=19)
    folium.TileLayer(tiles=maps.SATELLITE_TILES, attr="Esri World Imagery",
                     name="Спутник Esri", max_zoom=19).add_to(m)
    folium.raster_layers.ImageOverlay(
        image=rgba, bounds=bounds, opacity=1.0, name="Sentinel-2, %s" % month,
    ).add_to(m)
    folium.GeoJson(
        geometry, name="Контур поля",
        style_function=lambda _: {"color": "#FF2D55", "weight": 3, "fillOpacity": 0},
    ).add_to(m)
    folium.LayerControl(collapsed=False, position="topleft").add_to(m)

    if existing:
        note = "Загружено из labels/points_%s.csv: %d" % (field, len(existing))
        if skipped:
            note += " (строк не прочитано: %d)" % skipped
    else:
        note = "Файла labels/points_%s.csv пока нет: разметка с нуля." % field
    m.get_root().html.add_child(folium.Element(panel_html(field, month, note)))
    m.get_root().script.add_child(folium.Element(script_js(m.get_name(), field, month, existing)))
    return m


def save_label_map(field, month, path=None):
    """Читает снимок поля за месяц и сохраняет HTML. Возвращает путь к файлу."""
    month_number = int(month.split("-")[1])
    if month_number not in download.MONTHS:
        raise ValueError("месяц должен быть из списка: %s" % ", ".join(
            download.month_label(m) for m in download.MONTHS))
    geometry = download.read_geometry(download.FIELDS_DIR / (field + ".geojson"))
    scene = load.load_scene(field, month_number)
    existing, skipped = read_points(points_path(field))   # точки, поставленные раньше
    out = Path(path) if path else download.OUTPUT_DIR / ("label_%s.html" % field)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_label_map(field, geometry, scene, download.month_label(month_number), existing, skipped).save(str(out))
    return out


def main():
    parser = argparse.ArgumentParser(description="HTML-карта для ручной разметки точек")
    parser.add_argument("field", help="имя поля, например field_1")
    parser.add_argument("--month", default=download.month_label(7), help="месяц снимка: 2025-05, 2025-07 или 2025-08")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    try:
        out = save_label_map(args.field, args.month)
    except (ValueError, OSError) as e:
        raise SystemExit("Ошибка: %s" % e)
    print("Готово: %s" % out)
    existing, _ = read_points(points_path(args.field))
    if existing:
        print("Загружено точек из %s: %d (их можно дорисовывать)" % (points_path(args.field).name, len(existing)))
    print("Откройте файл в браузере, ставьте точки, нажмите «Скачать CSV» и положите файл в labels/points_%s.csv" % args.field)


if __name__ == "__main__":
    main()
