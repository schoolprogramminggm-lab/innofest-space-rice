"""Общие настройки тестов."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import zones


@pytest.fixture(autouse=True)
def no_model_by_default(tmp_path, monkeypatch):
    """
    По умолчанию тесты идут БЕЗ настоящей модели models/zones.joblib: зоны по пороговым правилам.
    Иначе результат тестов зависел бы от того, какая модель сейчас лежит в репозитории.
    Тесты классификатора кладут свою модель в этот же путь (zones.MODEL_PATH).
    """
    monkeypatch.setattr(zones, "MODEL_PATH", tmp_path / "models_none" / "zones.joblib")
