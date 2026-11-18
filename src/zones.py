"""
zones.py: деление поля на 4 зоны с помощью KMeans (искусственный интеллект).

KMeans сам собирает пиксели в 4 группы (кластера) по трём признакам: NDVI, SI, NDWI.
Но он даёт группам только номера 0, 1, 2, 3. Что это за зона, мы решаем по СРЕДНИМ
значениям индексов в группе: где NDVI самый высокий, там здоровый рис, и так далее.

В карте зон каждый пиксель хранит код: 0 = вне поля / нет данных, 1..4 = зона.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ключи зон. Код зоны в массиве = номер в списке + 1.
# not_rice (код 5): пиксель не рис по маске rice.py (серая зона на карте).
ZONE_KEYS = ["healthy", "stress", "salinity", "water", "not_rice"]
NOT_RICE_CODE = ZONE_KEYS.index("not_rice") + 1

# Названия для консоли
ZONE_NAMES_RU = {
    "healthy": "Здоровый рис",
    "stress": "Стресс",
    "salinity": "Засоление",
    "water": "Вода / пусто",
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
    Зоны с учётом маски риса (issue #19).

    «Здоровый рис» и «Стресс» считаются ТОЛЬКО по рисовым пикселям (rice == True):
    KMeans делит их на 2 группы, где NDVI выше, там здоровый рис, где ниже, там стресс.
    Все остальные пиксели поля это «не рисовое поле» (код 5).

    rice - True на рисовых пикселях (rice.rice_mask), mask - True внутри поля.
    Возвращает (zone_map, means): как find_zones, но в means только healthy и stress.
    Засоление и вода здесь пока не ищутся: такие пиксели в среднем не зелёные,
    и маска риса относит их к «не рисовым».
    """
    zone_map = np.zeros(ndvi.shape, dtype="uint8")
    zone_map[mask] = NOT_RICE_CODE

    # только рисовые пиксели, где все три индекса посчитались
    valid = rice & mask & np.isfinite(ndvi) & np.isfinite(si) & np.isfinite(ndwi)
    if valid.sum() < 2:
        return zone_map, {}   # риса нет: всё поле «не рисовое»

    features = np.column_stack([ndvi[valid], si[valid], ndwi[valid]])
    scaled = StandardScaler().fit_transform(features)
    cluster = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(scaled)

    # какая группа здоровая: та, где средний NDVI выше (NDVI это столбец 0)
    mean_ndvi = [features[cluster == c, 0].mean() if (cluster == c).any() else -9 for c in (0, 1)]
    healthy_cluster = int(np.argmax(mean_ndvi))
    codes = np.where(cluster == healthy_cluster, ZONE_KEYS.index("healthy") + 1,
                     ZONE_KEYS.index("stress") + 1).astype("uint8")
    zone_map[valid] = codes

    means = {}
    for key, c in (("healthy", healthy_cluster), ("stress", 1 - healthy_cluster)):
        if (cluster == c).any():
            means[key] = [float(v) for v in features[cluster == c].mean(axis=0)]
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
