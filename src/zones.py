"""
zones.py: деление поля на 4 зоны с помощью KMeans (искусственный интеллект).

KMeans сам собирает пиксели в 4 группы (кластера) по трём признакам: NDVI, SI, NDWI.
Но он даёт группам только номера 0, 1, 2, 3. Что это за зона, мы решаем по СРЕДНИМ
значениям индексов в группе: где NDVI самый высокий, там здоровый рис, и так далее.

В карте зон каждый пиксель хранит код: 0 = вне поля / нет данных, 1..4 = зона.
"""
import numpy as np
import rice as rice_rules   # пороги зон и затопления
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ключи зон. Код зоны в массиве = номер в списке + 1.
# empty: неудачный посев / пусто (внутри рисового массива, но растений нет).
# not_rice (код 6): пиксель вне рисового массива rice.py (серая зона на карте).
ZONE_KEYS = ["healthy", "stress", "salinity", "water", "empty", "not_rice"]
NOT_RICE_CODE = ZONE_KEYS.index("not_rice") + 1

# Названия для консоли
ZONE_NAMES_RU = {
    "healthy": "Здоровый рис",
    "stress": "Стресс",
    "salinity": "Засоление",
    "water": "Вода",
    "empty": "Неудачный посев / пусто",
    "not_rice": "Не рисовое поле",
}


def name_clusters(means):
    """
    По средним значениям кластеров решает, какая группа какая зона.

    means: список из 4 строк [NDVI, SI, NDWI], по одной на каждый кластер.
    Возвращает список из 4 ключей зон: names[номер_кластера] = "healthy" и т. д.

    Правила (каждая зона достаётся ровно одному кластеру):
      1. healthy  — самый высокий NDVI (много зелёных растений)
      2. stress   — второй по NDVI: растения есть, но слабее, чем у здорового
      3. из двух оставшихся (мало растений) смотрим на SI:
         salinity — где SI выше (белая соль на поверхности)
         water    — где SI ниже (вода или пустая земля без соли)

    Почему не по NDWI: летом воды в чеках нет, NDWI везде отрицательный, и «самый
    высокий NDWI» оказывался просто голой землёй. А вода всегда тёмная, SI у неё низкий.
    """
    if len(means) != 4:
        raise ValueError("Нужно ровно 4 кластера")

    # номера кластеров от самого высокого NDVI к самому низкому (индекс 0 = NDVI)
    by_ndvi = sorted(range(4), key=lambda c: means[c][0], reverse=True)

    names = [None] * 4
    names[by_ndvi[0]] = "healthy"
    names[by_ndvi[1]] = "stress"

    # два кластера с самым низким NDVI: у кого SI выше (индекс 1), тот засоление
    low_a, low_b = by_ndvi[2], by_ndvi[3]
    if means[low_a][1] >= means[low_b][1]:
        names[low_a], names[low_b] = "salinity", "water"
    else:
        names[low_a], names[low_b] = "water", "salinity"
    return names


def find_zones(ndvi, si, ndwi, mask):
    """
    Делит поле на 4 зоны.

    ndvi, si, ndwi — массивы индексов одного размера
    mask           — True там, где пиксель внутри поля

    Возвращает (zone_map, means):
      zone_map — массив uint8: 0 вне поля, 1 healthy, 2 stress, 3 salinity, 4 water
      means    — средние [NDVI, SI, NDWI] каждой зоны: {"healthy": [..], ...}
    """
    # Берём только пиксели внутри поля, где все три индекса посчитались (не NaN)
    valid = mask & np.isfinite(ndvi) & np.isfinite(si) & np.isfinite(ndwi)
    if valid.sum() < 4:
        raise ValueError("Слишком мало пикселей для 4 зон")

    # Таблица признаков: одна строка на пиксель, три столбца (NDVI, SI, NDWI)
    features = np.column_stack([ndvi[valid], si[valid], ndwi[valid]])

    # У индексов разный масштаб (SI около 0.1, NDVI от -1 до 1), поэтому приводим
    # все признаки к одинаковому масштабу, иначе SI почти не влиял бы на группы
    scaled = StandardScaler().fit_transform(features)

    # random_state=42, чтобы результат при каждом запуске был один и тот же
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    cluster = kmeans.fit_predict(scaled)  # номер кластера для каждого пикселя

    # Средние значения индексов в каждом кластере (в исходном масштабе)
    means = [features[cluster == c].mean(axis=0) for c in range(4)]
    names = name_clusters(means)

    # Собираем карту зон: код зоны = номер ключа + 1
    zone_map = np.zeros(ndvi.shape, dtype="uint8")
    codes = np.array([ZONE_KEYS.index(names[c]) + 1 for c in range(4)], dtype="uint8")
    zone_map[valid] = codes[cluster]

    zone_means = {names[c]: [float(v) for v in means[c]] for c in range(4)}
    return zone_map, zone_means


def find_rice_zones(ndvi, si, ndwi, rice, mask):
    """
    Зоны с учётом рисового массива (issue #19). Пороги: rice.py.

    rice - True на пикселях рисового массива (затоплены весной), mask - True внутри поля.
    Вне массива (но внутри поля) пиксель «не рисовое поле» (код 6).
    Внутри массива по состоянию в месяце снимка:
      NDVI > 0.6                         здоровый рис
      0.3 <= NDVI <= 0.6                 стресс
      NDVI < 0.3 и SI высокий            засоление
      NDVI < 0.3, SI низкий, NDWI > 0    вода
      NDVI < 0.3, остальное              неудачный посев / пусто
    Возвращает (zone_map, means): zone_map как в find_zones, means: средние [NDVI, SI, NDWI]
    по каждой зоне, где есть пиксели.
    """
    zone_map = np.zeros(ndvi.shape, dtype="uint8")
    zone_map[mask] = NOT_RICE_CODE

    # только пиксели массива, где все три индекса посчитались
    valid = rice & mask & np.isfinite(ndvi) & np.isfinite(si) & np.isfinite(ndwi)

    def code(key):
        return ZONE_KEYS.index(key) + 1

    low = valid & (ndvi < rice_rules.STRESS_NDVI)          # растений почти нет
    zone_map[valid & (ndvi > rice_rules.HEALTHY_NDVI)] = code("healthy")
    zone_map[valid & ~low & (ndvi <= rice_rules.HEALTHY_NDVI)] = code("stress")
    zone_map[low] = code("empty")                          # сначала «пусто», ниже уточняем
    zone_map[low & (si > rice_rules.SALT_SI)] = code("salinity")
    zone_map[low & (si <= rice_rules.SALT_SI) & (ndwi > rice_rules.WATER_NDWI)] = code("water")

    means = {}
    for key in ZONE_KEYS:
        pixels = zone_map == code(key)
        if key != "not_rice" and pixels.any():
            means[key] = [float(v[pixels].mean()) for v in (ndvi, si, ndwi)]
    return zone_map, means


def zone_areas(zone_map, pixel_ha):
    """
    Площадь каждой зоны: гектары и проценты от площади поля.
    Возвращает {"healthy": {"ha": 38.4, "pct": 71.0}, ...}
    """
    total_pixels = int((zone_map > 0).sum())
    result = {}
    for i, key in enumerate(ZONE_KEYS):
        pixels = int((zone_map == i + 1).sum())
        result[key] = {
            "ha": round(pixels * pixel_ha, 1),
            "pct": round(100 * pixels / total_pixels, 1) if total_pixels else 0.0,
        }
    return result
