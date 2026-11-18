"""Тесты export.py: JSON для AgroQyzyl. Интернет и снимки не нужны."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import export

# Обязательные ключи формата (docs/agroqyzyl-format.md)
KEYS = {"module", "version", "ok", "result", "message", "error"}
RESULT_KEYS = {"field", "date", "area_ha", "zones", "problem_area", "map_html", "map_png"}
ZONE_KEYS = {"healthy", "stress", "salinity", "water", "not_rice"}

# Маленький summary.json «как после run.py»: одно поле, один месяц
SUMMARY = {
    "field_2": {
        "2025-08": {
            "area_ha": 91.5,
            "zones": {
                "healthy": {"ha": 34.6, "pct": 37.9},
                "stress": {"ha": 36.8, "pct": 40.2},
                "salinity": {"ha": 1.8, "pct": 2.0},
                "water": {"ha": 18.3, "pct": 20.0},
                "not_rice": {"ha": 0.0, "pct": 0.0},
            },
            "rice_pct": 100.0,
            "flood_images": 5,
        }
    }
}


def check_envelope(answer):
    """Общая проверка: все ключи, оба языка, текст не пустой."""
    assert set(answer) == KEYS
    assert answer["module"] == "space_rice"
    assert answer["version"] == 1
    assert set(answer["message"]) == {"kk", "ru"}
    assert answer["message"]["kk"].strip() and answer["message"]["ru"].strip()


def test_ok_answer_has_all_keys_and_both_languages():
    answer = export.to_agroqyzyl("field_2", "2025-08", SUMMARY)
    check_envelope(answer)
    assert answer["ok"] is True
    assert answer["error"] is None

    result = answer["result"]
    assert set(result) == RESULT_KEYS
    assert result["field"] == "field_2"
    assert result["date"] == "2025-08"
    assert result["area_ha"] == 91.5
    assert set(result["zones"]) == ZONE_KEYS
    assert result["zones"]["stress"] == {"ha": 36.8, "pct": 40.2}
    assert result["problem_area"] is None          # место проблемы пока не определяем
    assert result["map_html"] == "output/map_field_2.html"
    assert result["map_png"] == "output/ndvi_field_2_2025-08.png"

    # в тексте есть цифры, десятичная запятая и главная проблема (стресс 40,2% > засоление 2,0%)
    assert "91,5" in answer["message"]["ru"] and "Главная проблема: стресс" in answer["message"]["ru"]
    assert "91,5" in answer["message"]["kk"] and "Негізгі мәселе: күйзеліс" in answer["message"]["kk"]


def test_main_problem_is_salinity_when_salt_is_bigger():
    summary = json.loads(json.dumps(SUMMARY))   # копия
    zones = summary["field_2"]["2025-08"]["zones"]
    zones["salinity"] = {"ha": 40.0, "pct": 43.7}
    answer = export.to_agroqyzyl("field_2", "2025-08", summary)
    assert "Главная проблема: засоление" in answer["message"]["ru"]
    assert "тұздану" in answer["message"]["kk"]


def test_no_data_for_month():
    answer = export.to_agroqyzyl("field_2", "2025-09", SUMMARY)
    check_envelope(answer)
    assert answer["ok"] is False
    assert answer["result"] is None
    assert "нет данных за 2025-09" in answer["error"]
    # пользователь видит понятный текст на обоих языках
    assert "2025-09" in answer["message"]["ru"]
    assert "2025-09" in answer["message"]["kk"]


def test_unknown_field():
    answer = export.to_agroqyzyl("field_9", "2025-08", SUMMARY)
    check_envelope(answer)
    assert answer["ok"] is False and answer["result"] is None
    assert "field_9" in answer["error"]


def test_no_summary_file(tmp_path):
    answer = export.to_agroqyzyl("field_2", "2025-08", summary_path=tmp_path / "нет.json")
    check_envelope(answer)
    assert answer["ok"] is False and answer["result"] is None
    assert "summary.json" in answer["error"]


def test_reads_summary_file(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(SUMMARY), encoding="utf-8")
    answer = export.to_agroqyzyl("field_2", "2025-08", summary_path=path)
    assert answer["ok"] is True


def test_save_writes_json_file(tmp_path):
    path = export.save_agroqyzyl("field_2", "2025-08", SUMMARY, out_dir=tmp_path)
    assert path.name == "field_2_2025-08.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == export.to_agroqyzyl("field_2", "2025-08", SUMMARY)


def test_example_file_matches_format():
    # examples/agroqyzyl_space_rice.json должен соответствовать формату
    example = json.loads((ROOT / "examples" / "agroqyzyl_space_rice.json").read_text(encoding="utf-8"))
    check_envelope(example)
    assert example["ok"] is True
    assert set(example["result"]) == RESULT_KEYS
    assert set(example["result"]["zones"]) == ZONE_KEYS


def test_not_rice_field_gives_ok_false_with_both_languages():
    summary = json.loads(json.dumps(SUMMARY))   # копия
    summary["field_2"]["2025-08"]["rice_pct"] = 12.5      # риса меньше 30%
    answer = export.to_agroqyzyl("field_2", "2025-08", summary)
    check_envelope(answer)
    assert answer["ok"] is False and answer["result"] is None
    assert "Здесь, похоже, не рисовое поле" in answer["message"]["ru"]
    assert "12,5" in answer["message"]["ru"] and answer["message"]["kk"].strip()
    assert answer["error"]

    summary["field_2"]["2025-08"]["rice_pct"] = 30.0      # ровно 30%: уже рисовое поле
    assert export.to_agroqyzyl("field_2", "2025-08", summary)["ok"] is True


def test_note_when_no_spring_images():
    summary = json.loads(json.dumps(SUMMARY))
    assert "Без проверки затопления" not in export.to_agroqyzyl("field_2", "2025-08", summary)["message"]["ru"]
    summary["field_2"]["2025-08"]["flood_images"] = 0
    answer = export.to_agroqyzyl("field_2", "2025-08", summary)
    assert answer["ok"] is True
    assert "Без проверки затопления" in answer["message"]["ru"] and answer["message"]["kk"].strip()
