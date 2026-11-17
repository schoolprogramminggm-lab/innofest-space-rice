"""
export.py: JSON для командного бота AgroQyzyl (формат: docs/agroqyzyl-format.md).

Берёт готовые площади зон из output/summary.json и собирает ответ:
    {"module": "space_rice", "version": 1, "ok": true/false,
     "result": {...}, "message": {"kk": "...", "ru": "..."}, "error": null | "текст"}

Если данных нет (нет месяца, нет поля, нет summary.json), функция НЕ падает,
а возвращает ok=false с понятным текстом: боту так проще, чем ловить ошибки.

Проверка: python src/export.py field_2 2025-08
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # корень проекта
OUTPUT_DIR = ROOT / "output"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

# Зоны в том же порядке, что в zones.ZONE_KEYS. Названия: (русский, казахский), как в легенде карты.
ZONES = [
    ("healthy", "Здоровый рис", "Сау күріш"),
    ("stress", "Стресс", "Күйзеліс"),
    ("salinity", "Засоление", "Тұзданған жер"),
    ("water", "Вода / пусто", "Су / бос жер"),
]


def num(value):
    """38.4 -> '38,4' (в русском и казахском десятичная запятая)."""
    return ("%.1f" % value).replace(".", ",")


def make_answer(ok, result, message, error):
    """Общая обёртка формата: все шесть ключей есть всегда."""
    return {
        "module": "space_rice",
        "version": 1,
        "ok": ok,
        "result": result,
        "message": message,
        "error": error,
    }


def make_error(error, message_kk, message_ru):
    """Ответ «данных нет»: ok=false, result=null, текст на двух языках."""
    return make_answer(False, None, {"kk": message_kk, "ru": message_ru}, error)


def load_summary(summary_path=None):
    """Читает summary.json. Если файла нет, возвращает None."""
    path = Path(summary_path) if summary_path else SUMMARY_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_message(field, month, area_ha, zones):
    """Короткий текст для пользователя на казахском и русском."""
    kk = ["%s алқабы, %s: %s га." % (field, month, num(area_ha))]
    ru = ["Поле %s, %s: %s га." % (field, month, num(area_ha))]
    for key, name_ru, name_kk in ZONES:
        kk.append("%s: %s га (%s%%)" % (name_kk, num(zones[key]["ha"]), num(zones[key]["pct"])))
        ru.append("%s: %s га (%s%%)" % (name_ru, num(zones[key]["ha"]), num(zones[key]["pct"])))

    # Главная проблема: что больше, стресс или засоление
    stress, salt = zones["stress"], zones["salinity"]
    if stress["pct"] >= salt["pct"]:
        kk.append("Негізгі мәселе: күйзеліс, %s%%." % num(stress["pct"]))
        ru.append("Главная проблема: стресс, %s%%." % num(stress["pct"]))
    else:
        kk.append("Негізгі мәселе: тұздану, %s%%." % num(salt["pct"]))
        ru.append("Главная проблема: засоление, %s%%." % num(salt["pct"]))
    return {"kk": "\n".join(kk), "ru": "\n".join(ru)}


def to_agroqyzyl(field, month, summary=None, summary_path=None):
    """
    Ответ для AgroQyzyl по одному полю и месяцу.

      field        - имя поля, например "field_2"
      month        - месяц в виде "ГГГГ-ММ", например "2025-08"
      summary      - уже прочитанный summary.json (словарь); если не дан, читаем файл
      summary_path - другой путь к summary.json (нужен тестам)
    """
    if summary is None:
        summary = load_summary(summary_path)
    if summary is None:
        return make_error(
            "нет файла summary.json, сначала запустите python run.py",
            "Алқап деректері әлі есептелмеген.",
            "Данные по полям ещё не посчитаны.",
        )

    if field not in summary:
        return make_error(
            "поля %s нет в summary.json" % field,
            "%s алқабы табылмады." % field,
            "Поле %s не найдено." % field,
        )

    scene = summary[field].get(month)
    if scene is None:
        return make_error(
            "нет данных за %s по полю %s" % (month, field),
            "%s алқабы үшін %s айының деректері жоқ." % (field, month),
            "Нет данных по полю %s за %s." % (field, month),
        )

    result = {
        "field": field,
        "date": month,
        "area_ha": scene["area_ha"],
        "zones": scene["zones"],
        # Место проблемы программа пока не определяет. Если оно появится в summary.json, подхватим.
        "problem_area": scene.get("problem_area"),
        "map_html": "output/map_%s.html" % field,
        "map_png": "output/ndvi_%s_%s.png" % (field, month),
    }
    message = build_message(field, month, scene["area_ha"], scene["zones"])
    return make_answer(True, result, message, None)


def save_agroqyzyl(field, month, summary=None, out_dir=None):
    """Пишет ответ в файл output/<поле>_<месяц>.json и возвращает путь к файлу."""
    answer = to_agroqyzyl(field, month, summary)
    folder = Path(out_dir) if out_dir else OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("%s_%s.json" % (field, month))
    path.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# Быстрая проверка: python src/export.py field_2 2025-08
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Как запускать: python src/export.py field_2 2025-08")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(to_agroqyzyl(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))
