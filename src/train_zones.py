"""
train_zones.py: обучение классификатора зон (issue #11).

Запуск одной командой:
    python src/train_zones.py

Что делает:
  1. читает ручную разметку labels/points_<поле>.csv (её ставит src/label_tool.py);
  2. для каждой точки берёт признаки её пикселя: NDVI, NDWI, SI, LSWI за май, июль и август
     и признак весеннего затопления (zones.FEATURE_NAMES, 13 чисел);
  3. делит точки на обучение и проверку 75/25 (со стратификацией: классы делятся поровну);
  4. обучает RandomForest из scikit-learn и сохраняет модель в models/zones.joblib;
  5. считает метрики на проверочных точках, которые модель не видела, и пишет их в консоль
     и в docs/zones-accuracy.md.

Нужны снимки в data/ (python src/download.py) и весенние файлы data/spring/ (их скачивает
python run.py). Интернет здесь не нужен.

Метрики не подгоняются: настройки модели заданы ниже один раз, и каким бы ни получился
результат, он записывается как есть.
"""
import sys
from collections import Counter
from datetime import date

import numpy as np
import sklearn
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import download
import label_tool
import load
import spring
import zones

REPORT_PATH = download.ROOT / "docs" / "zones-accuracy.md"
SPRING_DIR = download.DATA_DIR / "spring"   # весенние файлы полей (их скачивает run.py)

GOAL = 0.75          # цель issue #11: точность не ниже 75%
TEST_SHARE = 0.25    # доля точек для проверки
MIN_POINTS = 5       # меньше точек в классе: результат по нему ненадёжен
SEED = 42            # чтобы деление и модель при каждом запуске были одни и те же

SALT_NOTE = ("Класс «соль» (salinity) размечен по виду на снимке (белые и светлые пятна), "
             "без проверки на месте: наземных измерений засоления у нас нет.")


# ---------- Признаки полей ----------
def load_field(field):
    """
    Признаки всех пикселей поля. Возвращает словарь
    {"features": (высота, ширина, 13), "mask": True внутри поля, "transform", "crs"}.
    Если снимков нет, поднимает ValueError с понятным текстом.
    """
    by_month, scene = {}, None
    for month in zones.FEATURE_MONTHS:
        if not download.tif_path(field, month).exists():
            raise ValueError("нет снимка %s" % download.tif_path(field, month).name)
        scene = load.load_scene(field, month)
        by_month[month] = zones.scene_indices(scene)

    spring_tif = SPRING_DIR / ("%s_%s.tif" % (field, spring.CACHE_NAME))
    if not spring_tif.exists():
        raise ValueError("нет весеннего файла %s (его скачивает python run.py)" % spring_tif.name)
    shape = by_month[zones.FEATURE_MONTHS[0]]["ndvi"].shape
    flooded, images = spring.read(spring_tif, shape, scene["mask"])
    if images == 0:
        raise ValueError("нет безоблачных весенних снимков, признак затопления не посчитать")

    return {"features": zones.build_features(by_month, flooded), "mask": scene["mask"],
            "transform": scene["transform"], "crs": scene["crs"]}


# ---------- Признаки точек ----------
def collect_samples(points, fields):
    """
    Для каждой точки берёт признаки её пикселя.

      points - точки разметки (label_tool.read_points)
      fields - {поле: load_field(поле)}; если поля нет в словаре, его точки пропускаются

    Возвращает (X, y, groups, skipped): groups - поле каждой точки, skipped: {"поле": {причина: сколько точек}}.
    Пропускаются точки: вне контура поля (или вне снимка), без снимков поля, с пустыми
    признаками (NaN), дубли (два раза один пиксель, остаётся одна точка) и точки,
    где на один пиксель поставили разные классы (такой пиксель убираем целиком).
    """
    skipped = {}

    def skip(field, reason):
        skipped.setdefault(field, Counter())[reason] += 1

    pixels = {}   # (поле, строка, столбец) -> список классов точек, попавших в этот пиксель
    features_of = {}
    for p in points:
        data = fields.get(p["field"])
        if data is None:
            skip(p["field"], "нет снимков поля")
            continue
        xs, ys = warp_transform("EPSG:4326", data["crs"], [p["lon"]], [p["lat"]])
        row, col = (int(v[0]) for v in rowcol(data["transform"], xs, ys))
        height, width = data["mask"].shape
        if not (0 <= row < height and 0 <= col < width) or not data["mask"][row, col]:
            skip(p["field"], "вне контура поля")
            continue
        vector = data["features"][row, col]
        if not np.isfinite(vector).all():
            skip(p["field"], "нет данных в пикселе (NaN)")
            continue
        key = (p["field"], row, col)
        pixels.setdefault(key, []).append(p["class"])
        features_of[key] = vector

    X, y, groups = [], [], []
    for key, classes in pixels.items():
        if len(set(classes)) > 1:
            for _ in classes:
                skip(key[0], "разные классы на одном пикселе")
            continue
        for _ in classes[1:]:
            skip(key[0], "дубль (тот же пиксель)")
        X.append(features_of[key])
        y.append(classes[0])
        groups.append(key[0])
    return np.array(X, dtype="float32").reshape(-1, len(zones.FEATURE_NAMES)), np.array(y), np.array(groups), skipped


# ---------- Обучение и проверка ----------
def make_model(seed=SEED):
    """
    Классификатор. Настройки заданы один раз и под метрики не менялись.
    class_weight="balanced": редкие классы весят больше, иначе здоровый рис (самый частый класс) их заглушит.
    """
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)


def train_and_test(X, y, seed=SEED):
    """
    Делит точки 75/25 со стратификацией, обучает RandomForest, считает метрики на проверочных.
    Класс, в котором меньше двух точек, поделить нельзя: он в обучение не берётся (это попадёт в отчёт).

    Возвращает (модель, результат). Результат: словарь с accuracy, таблицей по классам,
    матрицей ошибок и числом точек каждого класса (всего, в обучении, в проверке).
    """
    counts = Counter(y)
    too_few = sorted(c for c, n in counts.items() if n < 2)
    keep = ~np.isin(y, too_few)
    if len(set(y[keep])) < 2:
        raise ValueError("для обучения нужно хотя бы два класса, в каждом не меньше двух точек")

    X_train, X_test, y_train, y_test = train_test_split(
        X[keep], y[keep], test_size=TEST_SHARE, stratify=y[keep], random_state=seed)

    model = make_model(seed)
    model.fit(X_train, y_train)

    classes = [c for c in zones.POINT_CLASSES if c in model.classes_]   # порядок как в списке классов
    predicted = model.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=classes, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "classes": classes,
        "precision": dict(zip(classes, precision)),
        "recall": dict(zip(classes, recall)),
        "f1": dict(zip(classes, f1)),
        "matrix": confusion_matrix(y_test, predicted, labels=classes),
        "total": {c: counts.get(c, 0) for c in zones.POINT_CLASSES},
        "train": Counter(y_train),
        "test": Counter(y_test),
        "too_few": too_few,
        # сколько точек класса лежит в рисовом массиве (признак затопления равен 1)
        "inside_rice": {c: int(X[y == c][:, -1].sum()) for c in zones.POINT_CLASSES},
    }
    return model, result


def leave_one_field_out(X, y, groups, seed=SEED):
    """
    Более строгая проверка: учим на двух полях, проверяем на третьем (его точек модель не видела,
    и соседних с ними тоже). Возвращает список {"field", "n", "accuracy", "unseen"};
    unseen: классы проверочного поля, которых не было в обучении (их модель предсказать не может).
    """
    found = []
    for field in sorted(set(groups)):
        test = groups == field
        if test.all() or len(set(y[~test])) < 2:
            continue
        model = make_model(seed).fit(X[~test], y[~test])
        found.append({
            "field": field, "n": int(test.sum()),
            "accuracy": float((model.predict(X[test]) == y[test]).mean()),
            "unseen": sorted(set(y[test]) - set(y[~test])),
        })
    return found


# ---------- Отчёт ----------
def report_text(result, skipped, n_points, model_path, model_kb, by_field=()):
    """
    Отчёт в Markdown: он печатается в консоль и пишется в docs/zones-accuracy.md.
    by_field: результат leave_one_field_out (может быть пустым).
    """
    acc = result["accuracy"]
    n_test = sum(result["test"].values())
    lines = ["# Точность классификатора зон", "",
             "Создано автоматически командой `python src/train_zones.py` (%s). Не править руками." % date.today().isoformat(),
             "", "## Итог", ""]

    if acc >= GOAL:
        lines.append("**Accuracy на отложенной выборке: %.1f%%** (%d точек). Цель issue #11 (≥ %d%%) достигнута по этой оценке; "
                     "ограничения ниже." % (100 * acc, n_test, 100 * GOAL))
    else:
        lines.append("**Accuracy на отложенной выборке: %.1f%%** (%d точек). **Цель issue #11 (≥ %d%%) НЕ достигнута.** "
                     "Результат записан как есть, настройки под него не подгонялись." % (100 * acc, n_test, 100 * GOAL))

    # честные предупреждения: пустые и малые классы
    # на поле, которого модель не видела, результат может быть хуже: об этом сразу в итоге
    worse = [r["field"] for r in by_field if r["accuracy"] < GOAL]
    if worse:
        lines += ["", "**Но на поле, которого модель не видела, точность ниже %d%%: %s (раздел «Проверка по полям»).**" % (
            100 * GOAL, ", ".join(worse))]

    lines += ["", "## Предупреждения", ""]
    warnings = []
    for c in zones.POINT_CLASSES:
        n = result["total"][c]
        if n == 0:
            warnings.append("Класс **%s**: 0 размеченных точек, классификатор его не знает и никогда не предскажет." % c)
        elif n < MIN_POINTS:
            warnings.append("Класс **%s**: всего %d точек (меньше %d), оценка по нему ненадёжна." % (c, n, MIN_POINTS))
        elif result["test"].get(c, 0) < MIN_POINTS:
            warnings.append("Класс **%s**: в проверке только %d точек (меньше %d), его F1 ненадёжен, "
                            "одна ошибка меняет его сильно." % (c, result["test"].get(c, 0), MIN_POINTS))
    for c in zones.POINT_CLASSES:
        # зоны считаются только внутри рисового массива: класс, которого там нет, на карте почти не появится
        if c != "not_rice" and result["total"][c] and result["inside_rice"][c] == 0:
            warnings.append("Класс **%s**: ни одна из %d точек не лежит в рисовом массиве (весной там воды не было). "
                            "Зоны считаются только внутри массива, поэтому на картах этот класс почти не появится." % (
                                c, result["total"][c]))
    for c in result["too_few"]:
        warnings.append("Класс **%s**: меньше двух точек, поделить на обучение и проверку нельзя, в обучение не взят." % c)
    warnings.append(SALT_NOTE)
    warnings.append("Деление на обучение и проверку случайное по точкам. Соседние точки одного поля похожи друг на друга, "
                    "поэтому на новых полях точность может быть ниже, чем здесь.")
    warnings.append("Метки ставил один человек по снимку июля 2025; точность считается относительно этих меток, "
                    "а не относительно реального состояния поля.")
    lines += ["- " + w for w in warnings]

    lines += ["", "## Точки по классам", "",
              "| Класс | Всего | В рисовом массиве | Обучение | Проверка | Precision | Recall | F1 |",
              "|---|---|---|---|---|---|---|---|"]
    for c in zones.POINT_CLASSES:
        if c in result["classes"]:
            lines.append("| %s | %d | %d | %d | %d | %.2f | %.2f | %.2f |" % (
                c, result["total"][c], result["inside_rice"][c], result["train"].get(c, 0), result["test"].get(c, 0),
                result["precision"][c], result["recall"][c], result["f1"][c]))
        else:
            lines.append("| %s | %d | %d | 0 | 0 | нет | нет | нет |" % (c, result["total"][c], result["inside_rice"][c]))
    macro = np.mean([result["f1"][c] for c in result["classes"]])
    lines += ["", "Средний F1 по классам, которые модель знает: %.2f." % macro]

    lines += ["", "## Матрица ошибок (проверочные точки)", "",
              "Строки: настоящий класс (по метке), столбцы: предсказание модели.", "",
              "| настоящий \\ предсказано | " + " | ".join(result["classes"]) + " |",
              "|---|" + "---|" * len(result["classes"])]
    for c, row in zip(result["classes"], result["matrix"]):
        lines.append("| %s | " % c + " | ".join(str(int(v)) for v in row) + " |")

    if by_field:
        lines += ["", "## Проверка по полям (строже)", "",
                  "Модель учится на двух полях и проверяется на третьем: так видно, как она работает на поле, "
                  "которого не знала. Это и есть главный вопрос для любого нового поля.", "",
                  "| Проверочное поле | Точек | Accuracy | Классов нет в обучении |", "|---|---|---|---|"]
        for r in by_field:
            lines.append("| %s | %d | %.1f%% | %s |" % (r["field"], r["n"], 100 * r["accuracy"], ", ".join(r["unseen"]) or "-"))
        if worse:
            lines += ["", "**Ниже %d%% на полях: %s.** На новых полях цель по этой оценке не достигается; "
                      "случайное деление выше оптимистично (соседние точки одного поля похожи)." % (100 * GOAL, ", ".join(worse))]

    lines += ["", "## Данные и модель", "",
              "- Точек разметки в файлах: %d, в обучение и проверку пошло: %d." % (n_points, sum(result["total"].values())),
              "- Признаки (%d): %s." % (len(zones.FEATURE_NAMES), ", ".join(zones.FEATURE_NAMES)),
              "- Классификатор: RandomForest (scikit-learn %s), 200 деревьев, class_weight=balanced, random_state=%d." % (
                  sklearn.__version__, SEED),
              "- Деление: %d%% обучение / %d%% проверка, со стратификацией по классам." % (
                  100 * (1 - TEST_SHARE), 100 * TEST_SHARE),
              "- Модель: `%s`, %.0f КБ." % (model_path, model_kb)]
    if model_kb >= 5 * 1024:
        lines.append("- **Модель больше 5 МБ: в git её не добавляем.**")

    lines += ["", "## Точки, не вошедшие в обучение", ""]
    if not skipped:
        lines.append("Все точки использованы.")
    for field, reasons in sorted(skipped.items()):
        lines.append("- %s: %s." % (field, ", ".join("%s: %d" % (r, n) for r, n in reasons.items())))
    return "\n".join(lines) + "\n"


def main():
    sys.stdout.reconfigure(errors="replace")

    # 1. точки разметки всех полей
    points = []
    for path in sorted(label_tool.LABELS_DIR.glob("points_field_*.csv")):
        found, bad = label_tool.read_points(path)
        print("%s: точек %d%s" % (path.name, len(found), ", не прочитано строк: %d" % bad if bad else ""))
        for p in found:
            p["field"] = p["field"] or path.stem.replace("points_", "")   # поле из имени файла, если столбец пуст
        points += found
    if not points:
        raise SystemExit("Нет точек: положите labels/points_<поле>.csv (python src/label_tool.py field_1)")

    # 2. признаки полей, для которых есть точки
    fields = {}
    for field in sorted({p["field"] for p in points}):
        try:
            fields[field] = load_field(field)
        except (ValueError, OSError) as e:
            print("%s: пропускаю, его точки не используются (%s)" % (field, e))

    # 3. признаки точек, обучение, метрики
    X, y, groups, skipped = collect_samples(points, fields)
    total_skipped = sum(sum(c.values()) for c in skipped.values())
    print("\nТочек в обучении и проверке: %d, пропущено: %d" % (len(y), total_skipped))
    for field, reasons in sorted(skipped.items()):
        print("  пропущено в %s: %s" % (field, ", ".join("%s: %d" % (r, n) for r, n in reasons.items())))
    try:
        model, result = train_and_test(X, y)
    except ValueError as e:
        raise SystemExit("Обучить не удалось: %s" % e)

    # 4. модель и отчёт
    model_path = zones.save_model(model)
    model_kb = model_path.stat().st_size / 1024
    text = report_text(result, skipped, len(points), model_path.relative_to(download.ROOT).as_posix(), model_kb,
                       leave_one_field_out(X, y, groups))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print("\n" + text)
    print("Модель: %s (%.0f КБ)%s" % (model_path, model_kb, "" if model_kb < 5 * 1024 else ", БОЛЬШЕ 5 МБ: в git не добавлять"))
    print("Отчёт: %s" % REPORT_PATH)


if __name__ == "__main__":
    main()
