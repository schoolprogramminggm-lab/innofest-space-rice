"""Тесты формул индексов (src/indices.py) на маленьких массивах. Интернет не нужен."""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import indices


def test_ndvi_known_values():
    # (B8 - B4) / (B8 + B4)
    assert np.isclose(indices.ndvi(b4=0.3, b8=0.5), 0.25)   # (0.5-0.3)/(0.5+0.3)
    assert np.isclose(indices.ndvi(b4=0.3, b8=0.3), 0.0)
    assert np.isclose(indices.ndvi(b4=0.1, b8=0.9), 0.8)    # густой рис
    assert indices.ndvi(b4=0.5, b8=0.1) < 0                 # вода/почва: отрицательный


def test_ndvi_division_by_zero_is_nan():
    # B8 + B4 = 0 -> NaN, без ошибки и без предупреждений
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # любое предупреждение NumPy стало бы ошибкой
        result = indices.ndvi(np.array([0.0, 0.3]), np.array([0.0, 0.5]))
    assert np.isnan(result[0])
    assert np.isclose(result[1], 0.25)


def test_ndvi_works_on_arrays():
    b4 = np.array([[0.1, 0.2], [0.3, 0.4]])
    b8 = np.array([[0.9, 0.2], [0.1, 0.4]])
    result = indices.ndvi(b4, b8)
    assert result.shape == (2, 2)
    assert np.isclose(result[0, 0], 0.8)
    assert np.isclose(result[0, 1], 0.0)
    assert result[1, 0] < 0


def test_si_known_values():
    # SI = sqrt(B2 * B4)
    assert np.isclose(indices.si(b2=0.04, b4=0.09), 0.06)
    assert np.isclose(indices.si(b2=0.25, b4=0.25), 0.25)
    assert indices.si(b2=0.0, b4=0.3) == 0


def test_si_negative_product_is_nan():
    # отрицательное отражение (шум) -> корень невозможен -> NaN, без предупреждений
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = indices.si(np.array([-0.01]), np.array([0.2]))
    assert np.isnan(result[0])


def test_ndwi_known_values():
    # NDWI = (B3 - B8) / (B3 + B8): вода больше 0
    assert np.isclose(indices.ndwi(b3=0.5, b8=0.3), 0.25)
    assert indices.ndwi(b3=0.1, b8=0.5) < 0     # растения: отрицательный
    assert indices.ndwi(b3=0.2, b8=0.05) > 0    # вода: положительный


def test_ndwi_division_by_zero_is_nan():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = indices.ndwi(np.array([0.0]), np.array([0.0]))
    assert np.isnan(result[0])
