"""
point.py: любое поле по запросу (issue #18): точка на карте или свой контур.

Что делает analyze_point(lat, lon, size, month):
  1. строит квадрат size x size метров вокруг точки (GeoJSON);
  2. берёт снимки Sentinel-2 из кэша data/cache/ или скачивает их (download.py, spring.py):
     июль и август (NDVI), нужный месяц и весну (был ли пиксель затоплен);
  3. строит маску «рисовое поле» (rice.py): затоплен весной И летом NDVI высокий;
  4. делит рисовые пиксели на зоны, рисует карту и картинку NDVI;
  5. возвращает ответ в формате AgroQyzyl (docs/agroqyzyl-format.md).
Если рисовых пикселей меньше 30%, ответ ok=false: «Здесь, похоже, не рисовое поле».

Функция ничего не печатает и не падает: если что-то не вышло, вернёт ok=false
и понятный текст на казахском и русском. Так её сможет вызвать бот.
Из командной строки: python run.py --point 44.9716,64.5267 --month 2025-07
"""
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv
from rasterio.warp import transform as warp_transform

import download
import export
import indices
import load
import maps
import rice
import spring
import zones

# ---------- Настройки ----------
CACHE_DIR = download.DATA_DIR / "cache"   # сюда складываем скачанные снимки
OUTPUT_DIR = download.OUTPUT_DIR          # карты, картинки и JSON (тесты подменяют эту папку)

DEFAULT_MONTH = "%d-07" % download.YEAR   # месяц по умолчанию: июль, рис в самом росте
MIN_SIZE, MAX_SIZE = 100, 3000            # размер квадрата, метров
SUMMER_MONTHS = [7, 8]                    # по их NDVI видно, вырос ли рис (rice.RICE_NDVI)
SEASON_MONTHS = range(5, 11)              # с мая по октябрь поля не убраны


# ---------- Разбор входных данных ----------
def check_point(lat, lon):
    """Проверяет, что широта и долгота имеют смысл. Если нет, поднимает ValueError."""
    # программа считает в UTM северного полушария, поэтому широта от 0 до 84
    if not (0 <= lat <= 84):
        raise ValueError("широта должна быть от 0 до 84, получено %s" % lat)
    if not (-180 <= lon <= 180):
        raise ValueError("долгота должна быть от -180 до 180, получено %s" % lon)


def parse_point(text):
    """'44.9716,64.5267' -> (44.9716, 64.5267). Первое число широта, второе долгота."""
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("точка задаётся двумя числами через запятую: широта,долгота")
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError("широта и долгота должны быть числами, получено %r" % text)
    check_point(lat, lon)
    return lat, lon


def parse_month(text):
    """'2025-07' -> 7. Работаем только с сезоном 2025 и месяцами из download.MONTHS."""
    found = re.fullmatch(r"(\d{4})-(\d{2})", text.strip())
    if not found:
        raise ValueError("месяц задаётся как ГГГГ-ММ, например 2025-07")
    year, month = int(found.group(1)), int(found.group(2))
    if year != download.YEAR or month not in download.MONTHS:
        names = ", ".join(download.month_label(m) for m in download.MONTHS)
        raise ValueError("доступны только месяцы сезона: %s" % names)
    return month


# ---------- Квадрат вокруг точки и кэш ----------
def square_geometry(lat, lon, size):
    """
    Квадрат size x size метров вокруг точки, как GeoJSON Polygon (градусы).
    Квадрат строим в метрах (UTM), потому что в градусах «500 м» разные по широте и долготе.
    """
    epsg = "EPSG:%d" % download.utm_epsg(lon)
    xs, ys = warp_transform("EPSG:4326", epsg, [lon], [lat])
    x, y = xs[0], ys[0]
    half = size / 2

    # четыре угла по кругу и замыкающая точка, равная первой
    corners_x = [x - half, x + half, x + half, x - half, x - half]
    corners_y = [y - half, y - half, y + half, y + half, y - half]
    lons, lats = warp_transform(epsg, "EPSG:4326", corners_x, corners_y)
    ring = [[round(a, 7), round(b, 7)] for a, b in zip(lons, lats)]
    return {"type": "Polygon", "coordinates": [ring]}


def cache_prefix(lat, lon, size):
    """Место и размер для имени файла: '44.9716_64.5267_500'. Координаты до 4 знаков (около 11 м)."""
    return "%.4f_%.4f_%d" % (lat, lon, size)


def cache_key(lat, lon, size, month_label):
    """Имя файла кэша без расширения: '44.9716_64.5267_500_2025-07'."""
    return "%s_%s" % (cache_prefix(lat, lon, size), month_label)


def cache_path(prefix, month_label):
    """data/cache/<lat>_<lon>_<size>_<месяц>.tif"""
    return CACHE_DIR / ("%s_%s.tif" % (prefix, month_label))


# ---------- Ответы с ошибкой (формат AgroQyzyl, ok=false) ----------
def bad_input_answer(problem):
    return export.make_error(
        "неверные входные данные: %s" % problem,
        "Сұраныс дұрыс емес. Координаталарды, мөлшерді және айды тексеріңіз.",
        "Неверный запрос. Проверьте координаты, размер и месяц.",
    )


def no_image_answer(month_label):
    return export.make_error(
        "нет безоблачного снимка за %s (облачность меньше %d%%)" % (month_label, download.MAX_CLOUDS),
        "%s айы үшін бұлтсыз Sentinel-2 суреті жоқ. Басқа айды таңдап көріңіз." % month_label,
        "За %s нет безоблачного снимка Sentinel-2. Попробуйте другой месяц." % month_label,
    )


def server_error_answer(problem):
    return export.make_error(
        "не удалось получить снимок: %s" % problem,
        "Серіктен сурет алу мүмкін болмады. Кейінірек қайталап көріңіз.",
        "Не удалось получить снимок со спутника. Попробуйте позже.",
    )


def season_note(today):
    """Если сейчас не сезон (поля убраны), возвращает пояснение {kk, ru}, иначе None."""
    if today.month in SEASON_MONTHS:
        return None
    return {
        "kk": "Қазір алқаптардағы егін жиналған, сондықтан %d жылғы маусымды көрсетеміз." % download.YEAR,
        "ru": "Сейчас поля убраны, поэтому показываем сезон %d года." % download.YEAR,
    }


# ---------- Снимок: кэш или скачивание ----------
def get_token(state):
    """Токен Copernicus. state: словарь, чтобы не входить дважды за один запрос."""
    if "token" not in state:
        load_dotenv(download.ROOT / ".env")
        state["token"] = download.get_token()    # ключи из .env, на экран не печатаются
    return state["token"]


def fetch_tif(geometry, month, tif, state):
    """
    Кладёт снимок в файл tif. Возвращает "ok" или "no_image" (нет безоблачного снимка).
    Если файл уже есть в кэше, ничего не скачиваем.
    state: словарь, где хранится токен, чтобы не входить в Copernicus дважды за один запрос.
    """
    if tif.exists():
        return "ok"
    body = download.build_request(geometry, month)
    if not download.download_tif(get_token(state), body, tif):
        raise requests.RequestException("сервер Copernicus вернул ошибку")
    if download.is_empty(tif):
        tif.unlink()                             # пустой файл в кэше не нужен
        return "no_image"
    return "ok"


def read_ndvi(name, geometry, month, tif):
    """Читает снимок из файла. Возвращает (сцена, NDVI)."""
    scene = load.load_scene(name, month, geometry=geometry, tif=tif)
    b = scene["bands"]
    return scene, indices.ndvi(b["B04"], b["B08"])


# ---------- Главная логика ----------
def analyze_geometry(name, geometry, month_label, prefix, today=None):
    """
    Полный конвейер для одного контура и месяца. Возвращает ответ AgroQyzyl.

      name   - имя для файлов результата (map_<name>.html и т. д.)
      prefix - начало имени файла в кэше
    """
    today = today or date.today()
    month = parse_month(month_label)
    state = {}
    try:
        # 1. Снимки: июль и август (NDVI для маски риса) и нужный месяц.
        #    Все идут через кэш: что уже скачано, второй раз не качаем.
        scenes = {}
        for m in dict.fromkeys(SUMMER_MONTHS + [month]):    # без повторов, порядок сохранён
            tif = cache_path(prefix, download.month_label(m))
            if fetch_tif(geometry, m, tif, state) == "ok":
                scenes[m] = read_ndvi(name, geometry, m, tif)
        if month not in scenes:
            return no_image_answer(month_label)
        summer = [scenes[m][1] for m in SUMMER_MONTHS if m in scenes]
        if not summer:
            return no_image_answer(download.month_label(SUMMER_MONTHS[0]))
        scene, ndvi = scenes[month]

        # 2. Весна: максимум NDWI по безоблачным снимкам 20 апреля - 20 июня (один файл на место)
        flood, flood_images = spring.flood(
            geometry, cache_path(prefix, "spring"), ndvi.shape, lambda: get_token(state))
    except SystemExit as e:                      # download.get_token: нет ключей или вход не удался
        return server_error_answer(str(e))
    except requests.RequestException as e:       # нет интернета, таймаут, ошибка сервера
        return server_error_answer(str(e))

    if scene["mask"].sum() < 4:                  # контур попал на место, где спутник не снимал
        return no_image_answer(month_label)

    # 3. Маска риса: затоплен весной И летом NDVI высокий (rice.py)
    mask = scene["mask"]
    rice_pixels = rice.rice_mask(rice.summer_ndvi(*summer), flood, mask)
    share = rice.rice_share(rice_pixels, mask)
    if not rice.is_rice_field(share):
        return export.not_rice_answer(100 * share)

    # 4. Зоны только по рисовым пикселям, остальное «не рисовое поле»
    b = scene["bands"]
    si = indices.si(b["B02"], b["B04"])
    ndwi = indices.ndwi(b["B03"], b["B08"])
    zone_map, _ = zones.find_rice_zones(ndvi, si, ndwi, rice_pixels, mask)
    areas = zones.zone_areas(zone_map, scene["pixel_ha"])
    field_ha = round(float(mask.sum()) * scene["pixel_ha"], 1)

    # 5. Картинка NDVI и карта
    indices.save_ndvi_png(ndvi, scene["mask"], "NDVI: %s, %s" % (name, month_label),
                          OUTPUT_DIR / ("ndvi_%s_%s.png" % (name, month_label)))
    maps.save_map(name, geometry, {month: (zone_map, scene["transform"], scene["crs"])},
                  OUTPUT_DIR / ("map_%s.html" % name))

    # 6. Ответ в формате AgroQyzyl: summary собираем в памяти, summary.json не трогаем
    summary = {name: {month_label: {"area_ha": field_ha, "zones": areas,
                                    "rice_pct": round(100 * share, 1), "flood_images": flood_images}}}
    answer = export.to_agroqyzyl(name, month_label, summary)
    note = season_note(today)
    if note:
        for lang in ("kk", "ru"):
            answer["message"][lang] += "\n" + note[lang]
    return answer


def analyze_point(lat, lon, size=500, month=DEFAULT_MONTH, today=None):
    """
    Главная функция для бота: квадрат size x size метров вокруг точки -> ответ AgroQyzyl.

      lat, lon - широта и долгота (градусы)
      size     - сторона квадрата в метрах (от 100 до 3000)
      month    - "ГГГГ-ММ" сезона 2025: 2025-05, 2025-07 или 2025-08
      today    - сегодняшняя дата (нужна только тестам)
    """
    try:
        check_point(lat, lon)
        if not (MIN_SIZE <= size <= MAX_SIZE):
            raise ValueError("размер квадрата от %d до %d метров, получено %s" % (MIN_SIZE, MAX_SIZE, size))
        parse_month(month)
    except ValueError as e:
        return bad_input_answer(e)

    size = int(size)
    geometry = square_geometry(lat, lon, size)
    name = "point_" + cache_prefix(lat, lon, size)
    return analyze_geometry(name, geometry, month.strip(), cache_prefix(lat, lon, size), today)


def analyze_geojson(path, month=DEFAULT_MONTH, today=None):
    """То же самое для своего контура из GeoJSON-файла."""
    try:
        parse_month(month)
        geometry = download.read_geometry(path)
        if geometry["type"] not in ("Polygon", "MultiPolygon"):
            raise ValueError("нужен контур (Polygon), а не %s" % geometry["type"])
    except (ValueError, KeyError, TypeError, OSError) as e:
        return bad_input_answer("контур %s не прочитан (%s)" % (path, e))

    # в имя кэша добавляем отпечаток контура: изменили контур, получили новый снимок
    digest = hashlib.md5(json.dumps(geometry, sort_keys=True).encode()).hexdigest()[:8]
    name = "geojson_" + Path(path).stem
    return analyze_geometry(name, geometry, month.strip(), "%s_%s" % (name, digest), today)
