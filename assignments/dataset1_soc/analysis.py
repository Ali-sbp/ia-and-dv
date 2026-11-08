"""Data and calculations for the Dataset 1 SOC checkpoint."""

from pathlib import Path
import os

import pandas as pd

HERE = Path(__file__).resolve().parent
NAME = "iavd_incidents_raw_target.parquet"
DEFAULT_DATA = Path(os.getenv("SOC_DATA", HERE / "data" / NAME))
SEP = "~~||~~"
SEVERITY = ("Критическая", "Высокая", "Средняя", "Низкая", "Отсутствует")
FIELDS = {
    "EntityId": "id",
    "Создан": "created",
    "Наименование": "rule",
    "Критичность": "severity",
    "Источник инцидента": "source",
    "Вердикт": "verdict",
    "Фаза реагирования": "phase",
}
ALL = "Все"
OPEN = "Открытые"
CLOSED = "Закрытые"


def load_cards(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден {path}. См. README.md: данные и запуск.")
    df = pd.read_parquet(path, columns=list(FIELDS)).rename(columns=FIELDS)
    if df.id.isna().any() or df.id.duplicated().any():
        raise ValueError("EntityId должен быть заполненным уникальным ключом.")
    df["created"] = pd.to_datetime(df.created, utc=True, errors="raise").dt.tz_localize(None)
    if df.created.isna().any():
        raise ValueError("В поле «Создан» есть пропуски.")
    df["day"] = df.created.dt.normalize()
    df["month"] = df.created.dt.to_period("M")
    df["severity"] = df.severity.fillna("Отсутствует")
    df["source"] = df.source.fillna("Не указан").str.replace(SEP, " + ", regex=False)
    df["rule"] = df.rule.fillna("Не указано").str.replace(SEP, " + ", regex=False)
    df["verdict"] = df.verdict.fillna("Не указан")
    # Workflow combinations are lists of tokens. Do not infer closure from the verdict.
    df["open"] = ~df.phase.fillna("").str.split(SEP, regex=False).map(lambda parts: "Закрыт" in parts)
    df["fp"] = df.verdict.eq("False Positive")
    df["unreviewed"] = df.verdict.eq("Не указан")
    return df.sort_values("created", ascending=False, kind="stable").reset_index(drop=True)


def subset(df, dates, severity=ALL, source=ALL, status=ALL):
    first, last = (pd.Timestamp(value).tz_localize(None).normalize() for value in dates)
    mask = df.day.between(first, last)
    if severity != ALL:
        mask &= df.severity.eq(severity)
    if source != ALL:
        mask &= df.source.eq(source)
    if status in (OPEN, CLOSED):
        mask &= df.open.eq(status == OPEN)
    return df.loc[mask]


def day_profile(df, start, end):
    """Mean cards per UTC calendar date, including dates with zero cards."""
    days = pd.date_range(start, end, freq="D")
    counts = df.groupby("day").size().reindex(days, fill_value=0)
    return {
        "Будни": (int(counts[days.dayofweek < 5].sum()), int((days.dayofweek < 5).sum())),
        "Выходные": (int(counts[days.dayofweek >= 5].sum()), int((days.dayofweek >= 5).sum())),
    }


def severity_counts(df):
    result = df.groupby("severity").agg(total=("id", "size"), fp=("fp", "sum"), unknown=("unreviewed", "sum"))
    order = [label for label in SEVERITY if label in result.index]
    order += sorted(set(result.index).difference(order))
    return result.reindex(order).assign(other=lambda t: t.total - t.fp - t.unknown)


def rule_counts(df):
    return df.groupby("rule").size().sort_index().sort_values(ascending=False, kind="stable")


def count(n):
    return f"{n:,}".replace(",", " ")


def decimal(n, digits=1):
    return f"{n:.{digits}f}".replace(".", ",")


def share(num, den):
    return f"{decimal(100 * num / den)} %" if den else "—"
