# Единый формат JSON для AgroQyzyl

Командный бот AgroQyzyl (модуль внутри бота RiceDoctor) получает ответы от двух программ:
**RiceDoctor** (диагноз листа по фото) и **SpaceRice** (карта поля со спутника).
Обе отдают JSON в одном формате, поэтому бот разбирает их одинаково.

Этот файл одинаковый в репозиториях `innofest-rice-doctor` и `innofest-space-rice`.

## Общая обёртка

```json
{
  "module": "rice_doctor",
  "version": 1,
  "ok": true,
  "result": {},
  "message": {"kk": "...", "ru": "..."},
  "error": null
}
```

| Ключ | Тип | Что значит |
|---|---|---|
| `module` | `"rice_doctor"` или `"space_rice"` | Кто сформировал ответ |
| `version` | число, сейчас `1` | Версия формата. Если формат изменится, число вырастет |
| `ok` | `true` / `false` | `true`: ответ получен. `false`: ответа нет, причина в `error` |
| `result` | объект или `null` | Данные ответа (см. ниже) |
| `message` | `{"kk": "...", "ru": "..."}` | Готовый текст для пользователя на двух языках. Есть **всегда**, и при `ok=false` тоже |
| `error` | `null` или текст | При `ok=true` всегда `null`. При `ok=false` понятная причина по-русски (для разработчика) |

Правила:
- Все шесть ключей есть в каждом ответе, даже если значение `null`.
- Бот может просто показать `message[язык пользователя]`, ничего не разбирая.
- Если `ok=false`, пользователю всё равно показывается `message`: там объяснение на его языке.

## `result` для `rice_doctor`

```json
{
  "disease": "blast",
  "name": {"kk": "Пирикуляриоз", "ru": "Пирикуляриоз"},
  "confidence": 0.91,
  "sure": true,
  "advice": {"kk": ["..."], "ru": ["..."]}
}
```

| Ключ | Тип | Что значит |
|---|---|---|
| `disease` | строка | Код класса: `healthy`, `blast`, `brown_spot`, `bacterial_blight` |
| `name` | `{kk, ru}` | Название болезни на двух языках |
| `confidence` | число от 0 до 1 | Уверенность нейросети (0.91 = 91%) |
| `sure` | `true` / `false` | `true`, если уверенность не ниже порога (по умолчанию 0.60) |
| `advice` | `{kk: [...], ru: [...]}` | Советы фермеру, списки строк. Если `sure=false`, списки пустые |

Когда `ok=false`:
- **«Не уверен»** (уверенность ниже порога): `result` **заполнен** лучшей догадкой сети, `sure=false`, `advice` пустой, в `error` написано, что уверенность ниже порога. Показывать пользователю нужно `message` («переснимите»), а не догадку.
- **Неизвестный класс или уверенность вне 0..1**: `result = null`.

## `result` для `space_rice`

```json
{
  "field": "field_2",
  "date": "2025-08",
  "area_ha": 91.5,
  "zones": {
    "healthy":  {"ha": 34.6, "pct": 37.9},
    "stress":   {"ha": 36.8, "pct": 40.2},
    "salinity": {"ha": 1.8,  "pct": 2.0},
    "water":    {"ha": 18.3, "pct": 20.0}
  },
  "problem_area": null,
  "map_html": "output/map_field_2.html",
  "map_png": "output/ndvi_field_2_2025-08.png"
}
```

| Ключ | Тип | Что значит |
|---|---|---|
| `field` | строка | Имя поля (`field_1`, `field_2`, `field_3`) |
| `date` | строка `ГГГГ-ММ` | Месяц снимка |
| `area_ha` | число | Площадь поля, га |
| `zones` | объект | Четыре зоны: `healthy` (здоровый рис), `stress` (стресс), `salinity` (засоление), `water` (вода / пусто). У каждой `ha` (гектары) и `pct` (процент от поля) |
| `problem_area` | строка или `null` | Где на поле проблема, например `"северо-восточный угол"`. Пока программа не определяет место, поэтому `null` |
| `map_html` | строка | Интерактивная карта поля (нужен интернет), путь от корня репозитория SpaceRice |
| `map_png` | строка | Картинка NDVI поля за этот месяц, путь от корня репозитория SpaceRice |

Когда `ok=false` (нет данных за месяц, поля нет в `summary.json`, самого `summary.json` нет): `result = null`.

## Примеры ответов с ошибкой

Нет данных за месяц (SpaceRice):

```json
{
  "module": "space_rice",
  "version": 1,
  "ok": false,
  "result": null,
  "message": {
    "kk": "field_2 алқабы үшін 2025-09 айының деректері жоқ.",
    "ru": "Нет данных по полю field_2 за 2025-09."
  },
  "error": "нет данных за 2025-09 по полю field_2"
}
```

Не уверен (RiceDoctor), коротко:

```json
{
  "module": "rice_doctor",
  "version": 1,
  "ok": false,
  "result": {"disease": "brown_spot", "confidence": 0.4, "sure": false, "...": "..."},
  "message": {"kk": "🤔 Сенімді емеспін. ...", "ru": "🤔 Не уверен. ..."},
  "error": "не уверен: уверенность 40% ниже порога 60%"
}
```

## Где лежит код и примеры

| | RiceDoctor | SpaceRice |
|---|---|---|
| Функция | `to_agroqyzyl(label, confidence)` в `bot/agroqyzyl.py` | `to_agroqyzyl(field, month)` в `src/export.py` |
| Пример ответа | `examples/agroqyzyl_rice_doctor.json` | `examples/agroqyzyl_space_rice.json` |
| Тесты | `tests/test_agroqyzyl.py` | `tests/test_export.py` |

SpaceRice при запуске `python run.py` сам пишет по одному файлу на поле и месяц: `output/<поле>_<месяц>.json`, например `output/field_2_2025-08.json`.
