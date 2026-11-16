"""
indices.py: индексы NDVI, SI, NDWI и картинка NDVI.

Каждая функция получает каналы как массивы NumPy (отражение от 0 до 1)
и считает индекс сразу для всех пикселей. Формулы: README, раздел 3.2.
"""
import matplotlib
matplotlib.use("Agg")  # рисуем в файл, окно не нужно
import matplotlib.pyplot as plt
import numpy as np


def _divide(top, bottom):
    """Деление массивов. Если делитель равен 0, получаем NaN (а не ошибку)."""
    top = np.asarray(top, dtype="float32")
    bottom = np.asarray(bottom, dtype="float32")
    result = np.full(np.broadcast(top, bottom).shape, np.nan, dtype="float32")
    # считаем только там, где делитель не ноль
    np.divide(top, bottom, out=result, where=(bottom != 0))
    return result


def ndvi(b4, b8):
    """NDVI = (B8 - B4) / (B8 + B4): зелёная масса растений. Рис в росте: 0.6-0.9."""
    b4 = np.asarray(b4, dtype="float32")
    b8 = np.asarray(b8, dtype="float32")
    return _divide(b8 - b4, b8 + b4)


def si(b2, b4):
    """SI = sqrt(B2 * B4): индекс засоления. Белая соль ярко отражает синий и красный свет."""
    b2 = np.asarray(b2, dtype="float32")
    b4 = np.asarray(b4, dtype="float32")
    product = b2 * b4
    result = np.full(product.shape, np.nan, dtype="float32")
    # корень из отрицательного числа невозможен: такие пиксели (шум) остаются NaN
    np.sqrt(product, out=result, where=(product >= 0))
    return result


def ndwi(b3, b8):
    """NDWI = (B3 - B8) / (B3 + B8): вода. Больше 0 значит вода."""
    b3 = np.asarray(b3, dtype="float32")
    b8 = np.asarray(b8, dtype="float32")
    return _divide(b3 - b8, b3 + b8)


def save_ndvi_png(ndvi_array, mask, title, path):
    """
    Рисует NDVI поля и сохраняет в PNG.
    Цветовая шкала RdYlGn: красный = мало растений, зелёный = много.
    Пиксели вне поля (mask == False) остаются белыми.
    """
    shown = np.where(mask, ndvi_array, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(shown, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title(title)       # подпись: поле и месяц
    ax.axis("off")
    fig.colorbar(image, ax=ax, label="NDVI", shrink=0.8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
