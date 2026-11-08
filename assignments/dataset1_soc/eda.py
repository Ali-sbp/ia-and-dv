"""Reproduce the 75 Dataset 1 EDA answers and three project hypotheses."""

import argparse
from collections import Counter
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis import DEFAULT_DATA, HERE, SEP, count, day_profile, decimal, load_cards, rule_counts, share

BANK = HERE.parents[1] / "materials/1.5 Пр Декомпозиция/iavd_eda_questions.md"
GROUPING = "Значение Инцидента (группировка)"
SOURCE_COLUMNS = [
    "EntityId", "Создан", "Вердикт", "Критичность", "verdict", GROUPING,
    "Фаза реагирования", "Статус инцидента из SIEM", "Спам", "Эскалировался на L2?",
    "Проверка на FP выполнена", "Ложное срабатывание", "Инцидент был изменен",
    "Группа устранения", "suser", "Полное имя", "Hostname Цели", "Ответственный",
    "Конвейер SIEM", "Время решения", "Период взятия в работу", "задержка перед автоназначением",
    "SLA по устранению Атаки", "SLA по взятию в работу Атаки", "processName",
    "Команда", "Пиктограмма IRP", "Теги", "Комментарий", "Ошибки", "duser", "Наименование",
]
TITLES = (
    "Структура данных", "Жизненный цикл", "Источники и нагрузка", "Время и SLA",
    "Процессы и мультизначные поля", "Качество данных", "Подготовка к ML",
)


def clean(value):
    return str(value).replace("|", "¦").replace("\n", " ")


def freq(s, n=None, anonymous=False):
    values = s.value_counts(dropna=False)
    if n is not None:
        values = values.head(n)
    return "; ".join(
        f"{('№ ' + str(i)) if anonymous else ('пропуск' if pd.isna(label) else clean(label))} — {count(int(value))}"
        for i, (label, value) in enumerate(values.items(), 1)
    )


def flags(s):
    return s.astype("string").str.casefold()


def pct_by(data, key, fp):
    totals = data.assign(_fp=fp).groupby(key, dropna=False)._fp.agg(["sum", "size"])
    return "; ".join(f"{clean(k) if pd.notna(k) else 'пропуск'} — {share(row['sum'], row['size'])}"
                     for k, row in totals.iterrows())


def schema_profile(parquet):
    names = parquet.schema_arrow.names
    nulls = Counter(dict.fromkeys(names, 0))
    for batch in parquet.iter_batches(batch_size=12_000):
        nulls.update({name: col.null_count for name, col in zip(names, batch.columns)})
    types = Counter()
    numbers = []
    for field in parquet.schema_arrow:
        dtype = field.type
        if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
            types["строковых"] += 1
        elif pa.types.is_timestamp(dtype):
            types["временных меток"] += 1
        elif pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
            types["числовых"] += 1
            numbers.append(field.name)
        else:
            types[str(dtype)] += 1
    return nulls, types, numbers


def answers(path, show_identifiers=False):
    pf = pq.ParquetFile(path)
    d = pd.read_parquet(path, columns=SOURCE_COLUMNS)
    n = len(d)
    nulls, types, number_names = schema_profile(pf)
    named = not show_identifiers
    id_duplicates = int(d.EntityId.duplicated().sum())
    verdict = d["Вердикт"]
    fp = verdict.eq("False Positive")
    flag = flags(d["Ложное срабатывание"]).eq("true").fillna(False)
    no_verdict = verdict.eq("Не указан")
    phase = d["Фаза реагирования"]
    combined = phase.str.contains(SEP, regex=False, na=False)
    phase_open = ~phase.fillna("").str.split(SEP, regex=False).map(lambda items: "Закрыт" in items)
    values = d[GROUPING].dropna()
    lengths = values.str.count(re.escape(SEP)).add(1)
    has_pair = values.str.contains("|", regex=False)
    elements = values.str.split(SEP, regex=False).explode()
    pairs = elements[elements.str.contains("|", regex=False)].str.split("|", n=1, expand=True)
    dates = d["Создан"].dt.tz_convert("UTC")
    months = dates.dt.tz_localize(None).dt.to_period("M")
    complete_months = months.value_counts().drop([months.min(), months.max()], errors="ignore")
    hours = dates.dt.hour.value_counts().reindex(range(24), fill_value=0)
    res = d["Время решения"]
    long_res = res.gt(30 * 86400)
    assigned = pd.to_datetime(d["задержка перед автоназначением"].median(), unit="s", utc=True)
    sla, sla_start = d["SLA по устранению Атаки"].median(), d["SLA по взятию в работу Атаки"].median()
    suser, person, target, analyst = (d[col].dropna() for col in ("suser", "Полное имя", "Hostname Цели", "Ответственный"))
    checks = flags(d["Проверка на FP выполнена"])
    gib = d["Группа устранения"].eq("ГИБ")
    case_variants = d["Hostname Цели"].dropna().drop_duplicates().to_frame("original")
    case_variants["key"] = case_variants.original.str.casefold()
    case_groups = case_variants.groupby("key").original.agg(list)
    case_groups = case_groups[case_groups.map(len).gt(1)]
    command_hits = d["Команда"].str.contains("powershell", case=False, regex=False, na=False)
    tags = d["Теги"].str.contains(SEP, regex=False, na=False)
    cross = pd.crosstab(verdict, d.verdict)
    one_to_one = cross.gt(0).sum(axis=0).eq(1).all() and cross.gt(0).sum(axis=1).eq(1).all()
    severity = d["Критичность"]
    out = {
        "1.1": f"{count(n)} строк × {len(pf.schema_arrow)} полей; строка соответствует карточке.",
        "1.2": "; ".join(f"{label} — {value}" for label, value in types.items()) + ". Дата без времени посчитана отдельно.",
        "1.3": "; ".join(f"{key} — {share(value, n)} пропусков" for key, value in nulls.most_common(3)) + ". Не использовать почти пустые поля как основу среза.",
        "1.4": f"{sum(value == 0 for value in nulls.values())} полей без NULL; это не проверка на пустые строки.",
        "1.5": f"{verdict.nunique()} категорий, включая явное «Не указан».",
        "1.6": f"{severity.nunique()} категорий; «Отсутствует» — значение выгрузки.",
        "1.7": f"{count(int(has_pair.sum()))} карточек, {share(int(has_pair.sum()), len(values))} от {count(len(values))} непустых. Одиночные элементы тоже встречаются.",
        "1.8": f"{count(int(lengths.max()))} элементов в фактическом поле «{GROUPING}». Колонки без суффикса нет; контроль 55 353 неприменим к этому полю.",
        "1.9": f"{id_duplicates} повторов; {int(d.EntityId.isna().sum())} пропусков ключа.",
        "1.10": f"{count(int(lengths.sum()))} строк из непустых списков, {count(int(lengths.sum()+d[GROUPING].isna().sum()))} с сохранением пропусков. Контроль ≈34,7 млн относится к другому полю/экспорту.",
        "1.11": f"`verdict`; взаимно однозначное соответствие с текстом по кросс-таблице: {'да' if one_to_one else 'нет'}. Для прогноза вердикта это утечка.",
        "2.1": freq(verdict) + ". Для вывода учитываем и карточки без вердикта.",
        "2.2": f"{count(int(fp.sum()))} / {count(n)} = {share(int(fp.sum()), n)}. Это доля меток аналитиков, не FPR модели.",
        "2.3": freq(phase) + ". Большинство записей в закрытой фазе.",
        "2.4": f"{count(int(combined.sum()))} сочетаний фаз по числу карточек; разделяем по `{clean(SEP)}`. Наличие «Закрыт» считаем закрытием, порядок переходов без временных меток не восстанавливаем.",
        "2.5": freq(phase[combined], 1) + ". Преобладает сочетание закрытия с анализом.",
        "2.6": freq(d["Статус инцидента из SIEM"]) + ". Этот статус не подменяет фазу SOAR.",
        "2.7": f"{count(int(flags(d['Спам']).eq('true').sum()))} карточек с отметкой True; это флаг, а не вердикт.",
        "2.8": f"{count(int(flags(d['Эскалировался на L2?']).eq('true').sum()))} отметок True. Не считаем NULL явным False.",
        "2.9": freq(checks) + ". Отдельно показываем неизвестное.",
        "2.10": f"Вердикт FP — {count(int(fp.sum()))}; флаг True — {count(int(flag.sum()))}; совпали — {count(int((fp & flag).sum()))}. Поля не взаимозаменяемы.",
        "2.11": f"{count(int(flags(d['Инцидент был изменен']).eq('true').sum()))} отметок True; это факт изменения, не его причина.",
        "2.12": freq(verdict[gib]) + f"; доля FP — {share(int((fp & gib).sum()), int(gib.sum()))}. Малую группу сравниваем с указанием объёма.",
        "3.1": f"{count(suser.nunique())} уникальных непустых записей; одна учётная запись не обязательно соответствует человеку.",
        "3.2": f"{count(person.nunique())} различных непустых имён, не гарантированно уникальных людей.",
        "3.3": freq(person, 3, anonymous=named) + ". Лидеры — по числу карточек с записанным именем.",
        "3.4": freq(suser, 3, anonymous=named) + ". Служебные учётные записи могут концентрировать события.",
        "3.5": f"{count(target.nunique())} различных заполненных значений; алиасы не объединялись.",
        "3.6": freq(target, 3, anonymous=named) + ". Частота упоминаний не равна числу атак на устройство.",
        "3.7": f"{analyst.nunique()} различных непустых значений ответственного.",
        "3.8": f"Медиана {decimal(analyst.value_counts().median(), 0)}; min {count(int(analyst.value_counts().min()))}; max {count(int(analyst.value_counts().max()))}; топ-3: {freq(analyst, 3, anonymous=named)}. Это объём карточек, не мера трудозатрат.",
        "3.9": freq(d["Группа устранения"]) + ". Основной поток приходится на L1.",
        "3.10": f"{d['Конвейер SIEM'].nunique()} заполненных категорий; топ-5: {freq(d['Конвейер SIEM'], 5, anonymous=named)}. Это поле не совпадает с продуктом-источником.",
        "4.1": f"{dates.min():%d.%m.%Y} — {dates.max():%d.%m.%Y} UTC; покрытие краёв неполное.",
        "4.2": f"{months.value_counts().idxmax()}: {count(int(months.value_counts().max()))} карточек.",
        "4.3": f"{complete_months.idxmin()}: {count(int(complete_months.min()))} карточек. Первый и последний месяцы исключены; контрольное значение банка не совпало с минимумом этой выгрузки.",
        "4.4": f"Пик {hours.idxmax():02}:00 ({count(int(hours.max()))}); минимум {hours.idxmin():02}:00 ({count(int(hours.min()))}), UTC.",
        "4.5": f"{share(int(dates.dt.dayofweek.ge(5).sum()), n)} карточек создано в выходные UTC; для сравнения нагрузки нормируем на число дней.",
        "4.6": f"{decimal(res.median()/3600, 2)} ч по заполненным значениям; {count(int(res.isna().sum()))} пропусков.",
        "4.7": f"{decimal(res.quantile(.9)/3600, 1)} ч, без очистки выбросов. Длинный хвост требует проверки.",
        "4.8": f"{share(int(long_res.sum()), n)} всех карточек; {share(int(long_res.sum()), int(res.notna().sum()))} среди записей с длительностью.",
        "4.9": "; ".join(f"{name} — {decimal(value/3600, 1)} ч" for name, value in d.groupby("Группа устранения")["Время решения"].median().items()) + ". Объёмы групп не сопоставимы.",
        "4.10": f"{decimal(d['Период взятия в работу'].median()/3600, 2)} ч по заполненным значениям.",
        "4.11": f"{count(int(d['задержка перед автоназначением'].median()))} соответствует {assigned:%d.%m.%Y %H:%M:%S} UTC как Unix-время в секундах. Это момент, а не длительность.",
        "4.12": f"Устранение {count(int(sla))} мс = {decimal(sla/3.6e6, 0)} ч; взятие {count(int(sla_start))} мс = {decimal(sla_start/60000, 0)} мин.",
        "5.1": freq(d["processName"].dropna(), 8) + ". Считаем исходные значения, без разбиения сочетаний.",
        "5.2": f"{share(int(command_hits.sum()), n)} всех карточек; {share(int(command_hits.sum()), int(d['Команда'].notna().sum()))} с заполненной командой. Поиск без учёта регистра.",
        "5.3": freq(d["Пиктограмма IRP"].dropna(), 5) + ". Категории годятся для группировки потока.",
        "5.4": freq(d["Теги"].dropna(), 3) + ". Сочетание считается одним значением в этом вопросе.",
        "5.5": f"{share(int(tags.sum()), n)} всех карточек; {share(int(tags.sum()), int(d['Теги'].notna().sum()))} среди заполненных.",
        "5.6": freq(pairs[1]) + ". Роли относятся к элементам списков, не к карточкам.",
        "5.7": freq(pairs[0], 3, anonymous=named) + ". Ключи сравниваются в исходном регистре.",
        "5.8": f"{count(len(pairs))} пар с разделителем; {count(pairs[0].nunique())} различных ключей. Остальные элементы — без пары.",
        "5.9": pct_by(d, "Критичность", fp) + ". Знаменатель — все карточки уровня, включая «Не указан».",
        "5.10": pct_by(d, "Группа устранения", fp) + ". Различия в составе потока не оценивают работу аналитиков.",
        "5.11": f"{count(int(d['Конвейер SIEM'].eq('srv-conv1.blinow-a12.iaivd.ru').sum()))} карточек данного конвейера; точное совпадение.",
        "5.12": f"По заполненному полю «{GROUPING}»: среднее {decimal(lengths.mean(), 2)}, медиана {decimal(lengths.median(), 0)}. Контроль из банка не соответствует этой колонке.",
        "6.1": f"{id_duplicates} дубликатов EntityId; число строк подходит как счётчик карточек.",
        "6.2": f"{count(int(d['Комментарий'].fillna('').str.strip().eq('').sum()))} NULL/пустых/пробельных; отдельно NULL: {count(int(d['Комментарий'].isna().sum()))}.",
        "6.3": f"{count(int(d['Ошибки'].notna().sum()))} заполненных; практически всегда пропуск.",
        "6.4": f"{count(int(d['Эскалировался на L2?'].isna().sum()))} NULL. В отчёте выводим отдельно от явных ответов.",
        "6.5": f"NULL — {count(int(verdict.isna().sum()))}; «Не указан» — {count(int(no_verdict.sum()))}.",
        "6.6": f"{count(len(case_groups))} групп имён хостов с вариантами регистра; пример: " +
               (" / ".join(map(clean, case_groups.iloc[0])) if show_identifiers else f"{len(case_groups.iloc[0])} варианта одного имени") +
               ". Нормализацию согласовать с владельцем данных.",
        "6.7": f"{count(round(res.max()/3600))} ч; {count(int(res.lt(0).sum()))} отрицательных длительностей. Максимум технически неправдоподобен; без валидации не используем для SLA.",
        "6.8": f"{count(int(sla))} мс = {decimal(sla/3.6e6, 0)} ч. Норматив в мс, длительность решения в секундах.",
        "7.1": f"FP — {count(int(fp.sum()))} ({share(int(fp.sum()), n)}); остальные — {count(int((~fp).sum()))} ({share(int((~fp).sum()), n)}). «Остальные» не равны подтверждённым атакам.",
        "7.2": f"Флаг «Ложное срабатывание»: {count(int(flag.sum()))} True, есть расхождения с вердиктом. Смысл целевой метки согласовать заранее.",
        "7.3": f"suser: {count(suser.nunique())}; duser: {count(d.duser.nunique())}; хосты: {count(target.nunique())}; наименования: {count(d['Наименование'].nunique())}. Учесть редкие категории.",
        "7.4": ", ".join(f"`{key}`" for key in number_names) + ". Числовой тип не делает идентификатор, код вердикта или постфактум-длительность допустимым признаком.",
        "7.5": f"{count(int(lengths.sum()))} строк из заполненных списков доступной колонки; для подсчёта достаточно числа разделителей, материализация всего explode не нужна.",
        "7.6": f"Флаг FP и код `verdict` раскрывают исход; {count(int((fp & flag).sum()))} совпадений FP и флага True. Не включать также тексты «Решение» и «Комментарий», созданные после триажа.",
        "7.7": "Преобразовать True/False в 1/0, NULL оставить отдельным состоянием. Использовать только флаги, доступные на момент прогноза.",
        "7.8": "Кандидаты: «Алерт», «Описание инцидента», «Команда», CMDLINE. Проверить время заполнения, удалить идентификаторы и обучать преобразования только на train.",
        "7.9": f"При случайной проверке использовать stratify=y, чтобы сохранить исходную долю FP {share(int(fp.sum()), n)}. Для проверки на будущих карточках — временное разбиение без перемешивания, с отчётом баланса по периодам.",
        "7.10": "Исключить EntityId, внутренние ID и код verdict; ключ хранить отдельно для связи. Сырые имена учётных записей и хостов проверять на утечки и переобучение.",
    }
    return out


def bank_titles():
    text = BANK.read_text().split("## Контрольные числа")[0]
    return {m[1]: m[2].strip().replace(r"\|", "¦") for m in re.finditer(
        r"^\| ([1-7]\.\d+) \| ((?:\\\||[^|])+)", text, flags=re.M
    )}


def render_answers(path, show_identifiers):
    calculated = answers(path, show_identifiers)
    titles = bank_titles()
    if len(calculated) != 75 or set(calculated) != set(titles):
        raise ValueError(f"Расхождение с банком: {set(calculated).symmetric_difference(titles)}")
    lines = [
        "# Датасет 1 — ответы на банк EDA (ML-трек)", "",
        "Источник: полный `iavd_incidents_raw_target.parquet`, версия практики 1.5. "
        "Расчёты выполнены в Python/Pandas на фактической выгрузке. "
        "Частоты приведены для строк карточек, временные срезы — в UTC.", "",
        "Персональные и инфраструктурные значения в рейтингах обозначены номерами; "
        "скрипт поддерживает вывод исходных подписей для локальной проверки.", "",
    ]
    for block, title in enumerate(TITLES, start=1):
        lines.extend([f"## {block}. {title}", "", "| № | Вопрос | Расчёт и интерпретация |", "| --- | --- | --- |"])
        lines.extend(f"| {key} | {titles[key]} | {value} |"
                     for key, value in calculated.items() if key.startswith(f"{block}."))
        lines.append("")
    return "\n".join(lines)


def render_hypotheses(path):
    df = load_cards(path)
    totals = df.groupby("severity").unreviewed.agg(["sum", "size"])
    window = df.day.min(), df.day.max()
    days = day_profile(df, *window)
    work_n, work_d = days["Будни"]
    rest_n, rest_d = days["Выходные"]
    a, b = work_n/work_d, rest_n/rest_d
    rules = rule_counts(df)
    total = len(df)
    rule_top = int(rules.head(10).sum())
    critical, medium = (totals.loc[key] for key in ("Критическая", "Средняя"))
    ratio = a/b
    gap = 100*(critical["sum"]/critical["size"]-medium["sum"]/medium["size"])
    return f"""# Датасет 1 — карточки СОК

## Данные и аудитория

Для дежурного аналитика и руководителя СОК: оценить нагрузку, выбрать правила для ревизии и увидеть сегменты, где не хватает вердиктов. Источник — полный учебный экспорт `iavd_incidents_raw_target.parquet` (практика 1.5), **{count(total)} карточек**, период создания **{window[0]:%d.%m.%Y}–{window[1]:%d.%m.%Y} UTC**. Один `EntityId` = одна карточка; в неё может входить несколько алертов. Проверены заполненность ключа, уникальность карточек и читаемость даты. В срезе по умолчанию **{count(int(df.open.sum()))} открытых** карточек (фаза не содержит токена «Закрыт»), **{count(int(df.unreviewed.sum()))}** без указанного вердикта и **{count(int(df.fp.sum()))}** с вердиктом False Positive.

## Что проверили

| Гипотеза | Поля и график | Результат на полной выгрузке |
| --- | --- | --- |
| В будни на календарный день приходится больше карточек, чем в выходные. | `Создан`; два столбца среднего дневного объёма. | Будни: {count(work_n)} / {work_d} = **{decimal(a)}**; выходные: {count(rest_n)} / {rest_d} = **{decimal(b)}**. Отношение **{decimal(ratio, 2)}×**, гипотеза {"поддержана" if ratio>1 else "не поддержана"}. |
| Десять наиболее частых названий правил дают не меньше 30% карточек. | `Наименование`; горизонтальный топ-10 по числу карточек. | **{count(rule_top)} / {count(total)} = {share(rule_top, total)}**, гипотеза {"поддержана" if rule_top/total>=.3 else "не поддержана"}. Первое правило — `{rules.index[0]}`, {count(int(rules.iloc[0]))} карточек. |
| У критических карточек доля «Не указан» выше, чем у средней критичности. | `Критичность`, `Вердикт`; 100%-ные составные столбцы. | Критические: **{share(critical["sum"], critical["size"])}** ({count(int(critical["sum"]))} из {count(int(critical["size"]))}); средние: **{share(medium["sum"], medium["size"])}** ({count(int(medium["sum"]))} из {count(int(medium["size"]))}). Разница **{decimal(gap)} п.п.**, гипотеза {"поддержана" if gap>0 else "не поддержана"}. |

## Выводы для работы

- Для планирования очереди сравниваем объём в расчёте на календарный день. Праздники отдельно не размечены; первое и последнее число выгрузки покрыты частично. График по месяцам показывает изменения потока, крайние месяцы не сравниваем как полные.
- Топ-10 даёт список кандидатов для ревизии правил. Большой объём сам по себе не означает плохое правило: сначала проверить долю ложных срабатываний, состав трафика и последствия изменения.
- Высокая доля неопределённых вердиктов у критических карточек — повод проверить заполнение и порядок закрытия. Она не говорит, сколько атак пропустила модель. Знаменатель долей — все карточки выбранной группы; на малых срезах учитываем объём.
- Для датасета 1 **False Positive — вердикт аналитика**. Оценка ошибок модели потребует предсказаний и согласованных истинных меток (датасет 2).

Приложение даёт фильтры по датам создания, критичности, источнику и состоянию. Все показатели и графики используют один срез. Составные названия и источники остаются одним значением карточки, поэтому запись не дублируется при подсчёте. Таблица показывает не более 200 самых свежих карточек среза. [Расчёты по 75 вопросам](EDA_1_5.md) и [запуск](README.md).
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--show-identifiers", action="store_true", help="Включить исходные подписи в локальный отчёт.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "EDA_1_5.md").write_text(render_answers(args.data, args.show_identifiers), encoding="utf-8")
    (args.output_dir / "ОТЧЁТ.md").write_text(render_hypotheses(args.data), encoding="utf-8")
