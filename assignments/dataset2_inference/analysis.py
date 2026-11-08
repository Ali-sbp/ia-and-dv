"""Evaluation of the supplied SOC inference log; no model fitting or scoring."""

from dataclasses import dataclass
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path(os.getenv("SOC_INFERENCE_DATA", HERE / "data"))
PREDICTIONS = "MODEL_PREDICTIONS_target.parquet"
CARDS = "iavd_incidents_raw3_target.parquet"
BASELINE = 0.18
DEFAULT_COST = 4


def integer(n):
    return f"{n:,}".replace(",", " ")


def decimal(n, digits=2):
    return f"{n:.{digits}f}".replace(".", ",") if n is not None else "—"


def percent(n):
    return decimal(n * 100, 1) + " %" if n is not None else "—"


def cutoff(n):
    return f"{n:.8f}".rstrip("0").rstrip(".").replace(".", ",")


@dataclass(frozen=True)
class Point:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def alerts(self):
        return self.tp + self.fp

    @property
    def precision(self):
        return self.tp / self.alerts if self.alerts else None

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    def cost(self, k):
        return k * self.fn + self.fp

    def fbeta(self, k):
        # beta squared is k. Algebra avoids division by undefined precision.
        return (1 + k) * self.tp / ((1 + k) * self.tp + k * self.fn + self.fp)


class Sweep:
    """Compute threshold counts using cumulative counts of sorted actual labels."""

    def __init__(self, y, scores):
        y = np.asarray(y)
        scores = np.asarray(scores, dtype=float)
        if y.shape != scores.shape or y.ndim != 1 or set(np.unique(y)) != {0, 1}:
            raise ValueError("Нужны согласованные массивы меток обоих классов (0 и 1).")
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Оценки модели должны быть в диапазоне [0, 1].")
        order = np.argsort(scores, kind="stable")
        self.scores = scores[order]
        self.positives = np.r_[0, np.cumsum(y[order], dtype=int)]
        self.candidates = np.union1d(np.unique(self.scores), [0.0, 1.0])
        self._grid = self.counts(self.candidates)

    def counts(self, thresholds):
        # searchsorted(left) puts equal scores on the positive side of the cutoff.
        i = np.searchsorted(self.scores, thresholds, side="left")
        fn = self.positives[i]
        tn = i - fn
        return self.positives[-1] - fn, len(self.scores) - self.positives[-1] - tn, fn, tn

    def at(self, threshold):
        return Point(float(threshold), *(int(value) for value in self.counts(threshold)))

    def best(self, k):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k должен быть положительным целым числом.")
        tp, fp, fn, _ = self._grid
        costs = k * fn + fp
        min_cost = int(np.flatnonzero(costs == costs.min())[-1])

        # Compare rational F scores exactly, so tie-breaking is deterministic.
        num = (1 + k) * tp
        den = num + k * fn + fp
        index = int(np.argmax(num / den))
        tied = np.flatnonzero(num * den[index] == num[index] * den)
        max_f = int(tied[-1])
        return self.at(self.candidates[min_cost]), self.at(self.candidates[max_f])


@dataclass(frozen=True)
class Log:
    labeled: pd.DataFrame
    total: int
    missing_label: int
    excluded_label: int
    no_matching_card: int
    sweep: Sweep

    def error_examples(self, threshold, kind, limit=10):
        actual = 0 if kind == "FP" else 1
        detected = self.labeled.score.ge(threshold)
        error = self.labeled.loc[self.labeled.y.eq(actual) & detected.eq(kind == "FP")]
        # Closest to the boundary first; ties sorted by card ID.
        ranked = (error.assign(distance=(error.score - threshold).abs())
                  .sort_values(["distance", "incident_id"]).head(limit))
        frame = ranked[["incident_id", "score", "severity", "rule"]].copy()
        frame["score"] = frame.score.map(lambda x: decimal(x, 5))
        frame["actual"] = "Легитимная активность" if actual == 0 else "Инцидент"
        frame["predicted"] = "Инцидент" if actual == 0 else "Легитимная активность"
        return frame.rename(columns=dict(incident_id="Карточка", score="Оценка", severity="Критичность",
                                        rule="Правило", actual="Разметка", predicted="Прогноз"))


def rule_label(raw):
    raw = str(raw)
    if raw.startswith("["):
        try:
            names = json.loads(raw)
            if isinstance(names, list):
                raw = " + ".join(str(name) for name in names if name)
        except json.JSONDecodeError:
            pass
    return (raw or "Не указано")[:120]


def load(data_dir=DATA):
    p = Path(data_dir) / PREDICTIONS
    c = Path(data_dir) / CARDS
    for file in (p, c):
        if not file.is_file():
            raise FileNotFoundError(f"Не найден {file}. См. README.md: загрузка данных.")
    predictions = pd.read_parquet(p, columns=["incident_id", "y", "y_pred_proba", "Flag_ValidLabel"])
    cards = pd.read_parquet(c, columns=["incident_id", "Критичность", "Правило корреляции SIEM"])
    for name, table in (("лог", predictions), ("карточки", cards)):
        if table.incident_id.isna().any() or table.incident_id.duplicated().any():
            raise ValueError(f"{name}: incident_id должен быть уникальным и заполненным.")
    valid = predictions.y.isin([0, 1])
    if not predictions.Flag_ValidLabel.eq(1).equals(valid):
        raise ValueError("Flag_ValidLabel не согласуется с бинарными метками.")
    joined = predictions.loc[valid, ["incident_id", "y", "y_pred_proba"]].merge(
        cards, on="incident_id", how="left", validate="one_to_one"
    ).rename(columns={"y_pred_proba": "score", "Критичность": "severity", "Правило корреляции SIEM": "rule"})
    joined["severity"] = joined.severity.fillna("Не указана")
    joined["rule"] = joined.rule.fillna("Не указано").map(rule_label)
    return Log(
        joined, len(predictions), int(predictions.y.isna().sum()),
        int((~valid & predictions.y.notna()).sum()),
        int((~predictions.incident_id.isin(cards.incident_id)).sum()),
        Sweep(joined.y, joined.score),
    )
