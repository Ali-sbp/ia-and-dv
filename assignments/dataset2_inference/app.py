"""Interactive Taipy dashboard for the Dataset 2 threshold exercise."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from taipy.gui import Gui, State
from taipy.gui import builder as tgb

from analysis import BASELINE, DATA, DEFAULT_COST, cutoff, decimal, integer, load, percent

BLUE, AMBER, INK = "#2855b6", "#a14c0e", "#18263d"
GRID = "#e3eaf4"
CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}
COMPARE_COLUMNS = ["Режим", "Порог", "FP", "FN", "Fβ", "Стоимость"]
ERROR_COLUMNS = ["Карточка", "Оценка", "Критичность", "Правило", "Разметка", "Прогноз"]
ERROR_LABELS = ["FP · ложная тревога", "FN · пропуск инцидента"]


def tradeoff_figure(sweep, chosen, optimum, k, is_cost):
    x = np.unique(np.r_[np.linspace(0, 1, 501), chosen.threshold, optimum.threshold, BASELINE])
    tp, fp, fn, _ = sweep.counts(x)
    y = k * fn + fp if is_cost else (1 + k) * tp / ((1 + k) * tp + k * fn + fp)
    unit = "Стоимость" if is_cost else "Fβ"
    y_at = lambda p: p.cost(k) if is_cost else p.fbeta(k)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name=unit, line=dict(color=BLUE, width=2.5),
                               hovertemplate="Порог %{x:.5f}<br>" + unit + ": %{y:.4~f}<extra></extra>"))
    for p, label, color, symbol in (
        (optimum, "Оптимум", AMBER, "diamond"), (chosen, "Текущий порог", INK, "circle-open")
    ):
        fig.add_scatter(x=[p.threshold], y=[y_at(p)], mode="markers", name=label,
                        marker=dict(color=color, size=11, symbol=symbol, line=dict(width=2, color=color)),
                        hovertemplate=label + "<br>Порог %{x:.5f}<br>" + unit + ": %{y:.4~f}<extra></extra>")
    fig.update_layout(
        height=260, showlegend=False, margin=dict(l=48, r=16, t=15, b=38),
        font=dict(family="system-ui, sans-serif", size=12, color=INK),
        paper_bgcolor="white", plot_bgcolor="white", hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(range=[0, 1], title="Порог τ", fixedrange=True, showgrid=False, automargin=True)
    fig.update_yaxes(fixedrange=True, gridcolor=GRID, zeroline=False, automargin=True, rangemode="tozero")
    return fig


def view_values(log, threshold, k, error_label):
    chosen = log.sweep.at(threshold)
    baseline = log.sweep.at(BASELINE)
    min_cost, max_f = log.sweep.best(k)
    comparison = pd.DataFrame([
        (title, cutoff(p.threshold), p.fp, p.fn, decimal(p.fbeta(k), 4), p.cost(k))
        for title, p in (("Учебный", baseline), ("Минимум стоимости", min_cost), ("Максимум Fβ", max_f))
    ], columns=COMPARE_COLUMNS)
    kind = error_label[:2]
    errors = log.error_examples(threshold, kind)
    total_errors = chosen.fp if kind == "FP" else chosen.fn
    return {
        "threshold_text": f"Порог τ = {cutoff(threshold)}",
        "cost_text": f"Цена пропуска k = {k}",
        "precision_text": percent(chosen.precision),
        "recall_text": percent(chosen.recall),
        "cost_value": integer(chosen.cost(k)),
        "fbeta_text": decimal(chosen.fbeta(k), 4),
        "tp_text": integer(chosen.tp),
        "fp_text": integer(chosen.fp),
        "fn_text": integer(chosen.fn),
        "tn_text": integer(chosen.tn),
        "alert_text": f"При текущем пороге: {integer(chosen.alerts)} тревог из {integer(len(log.labeled))} размеченных записей.",
        "comparison_text": (f"β = √{k} ≈ {decimal(math.sqrt(k), 3)}. "
                            f"Порог min Cost: {cutoff(min_cost.threshold)}; порог max Fβ: {cutoff(max_f.threshold)}."),
        "comparison": comparison,
        "cost_fig": tradeoff_figure(log.sweep, chosen, min_cost, k, True),
        "f_fig": tradeoff_figure(log.sweep, chosen, max_f, k, False),
        "examples": errors,
        "examples_note": (f"Показано {integer(len(errors))} из {integer(total_errors)} ошибок, ближайших к порогу."
                          if total_errors else "Ошибок выбранного типа при текущем пороге нет."),
    }


def refresh(state: State):
    for key, value in view_values(log, float(state.threshold), int(state.k), state.error_label).items():
        state.assign(key, value)


def change_controls(state: State, _name, _value):
    refresh(state)


def change_examples(state: State, _name, _value):
    kind = state.error_label[:2]
    p = log.sweep.at(float(state.threshold))
    examples = log.error_examples(p.threshold, kind)
    total = p.fp if kind == "FP" else p.fn
    state.examples = examples
    state.examples_note = (f"Показано {integer(len(examples))} из {integer(total)} ошибок, ближайших к порогу."
                           if total else "Ошибок выбранного типа при текущем пороге нет.")


def use_baseline(state: State):
    state.threshold = BASELINE
    refresh(state)


def use_min_cost(state: State):
    state.threshold = log.sweep.best(int(state.k))[0].threshold
    refresh(state)


def use_max_f(state: State):
    state.threshold = log.sweep.best(int(state.k))[1].threshold
    refresh(state)


def metric(label, value, note, ident):
    with tgb.part(class_name="metric", id=ident):
        tgb.text(label, class_name="metric-label")
        tgb.text(value, class_name="metric-value")
        tgb.text(note, class_name="muted")


def matrix_cell(code, label, value, error=False):
    with tgb.part(class_name="cell error" if error else "cell", id="cell-" + code.lower()):
        tgb.text(code + " · " + label, class_name="cell-label")
        tgb.text(value, class_name="cell-count")


def build_page():
    with tgb.Page() as page:
        with tgb.part(class_name="soc-shell"):
            with tgb.part(class_name="hero"):
                tgb.text("ИАиВД / ML-ИНЖИНИРИНГ / ДАТАСЕТ 02", class_name="eyebrow")
                tgb.text("# Инференс модели СОК", mode="md")
                tgb.text("Сравните ложные тревоги, пропуски и стоимость ошибок при разных порогах.",
                         class_name="hero-copy")
                tgb.text("{dataset_caption}", class_name="hero-meta")
            with tgb.part(class_name="panel"):
                tgb.text("## Настройка порога", mode="md")
                with tgb.part(class_name="controls"):
                    with tgb.part(class_name="control"):
                        tgb.text("{threshold_text}", class_name="metric-label", id="current-threshold")
                        tgb.slider("{threshold}", min=0, max=1, step=.001, id="threshold-control",
                                   on_change=change_controls, width="100%")
                        tgb.text("Карточка считается инцидентом, если её оценка ≥ τ.", class_name="muted")
                    with tgb.part(class_name="control"):
                        tgb.text("{cost_text}", class_name="metric-label")
                        tgb.slider("{k}", min=1, max=20, step=1, id="cost-control",
                                   on_change=change_controls, width="100%")
                        tgb.text("Одна ложная тревога стоит 1, один пропуск — k.", class_name="muted")
                with tgb.part(class_name="presets"):
                    tgb.button("Учебный 0,18", on_action=use_baseline)
                    tgb.button("Минимум стоимости", on_action=use_min_cost)
                    tgb.button("Максимум Fβ", on_action=use_max_f)
                tgb.text("{alert_text}", class_name="muted alert-count", id="alert-count")
            with tgb.part(class_name="metric-grid"):
                metric("Точность (precision)", "{precision_text}", "Доля верных среди тревог", "precision-metric")
                metric("Полнота (recall)", "{recall_text}", "Доля найденных инцидентов", "recall-metric")
                metric("Стоимость", "{cost_value}", "k × FN + FP", "cost-metric")
                metric("Fβ", "{fbeta_text}", "β = √k", "fbeta-metric")
            with tgb.part(class_name="primary-grid"):
                with tgb.part(class_name="panel"):
                    tgb.text("## Матрица ошибок", mode="md")
                    tgb.text("Строки — фактический класс; столбцы — прогноз.", class_name="muted")
                    with tgb.part(class_name="matrix"):
                        tgb.text("Факт ↓ / Прогноз →", class_name="matrix-head")
                        tgb.text("Инцидент", class_name="matrix-head")
                        tgb.text("Легитимно", class_name="matrix-head")
                        tgb.text("Инцидент", class_name="matrix-head")
                        matrix_cell("TP", "верно найден", "{tp_text}")
                        matrix_cell("FN", "пропуск", "{fn_text}", True)
                        tgb.text("Легитимно", class_name="matrix-head")
                        matrix_cell("FP", "ложная тревога", "{fp_text}", True)
                        matrix_cell("TN", "верно исключён", "{tn_text}")
                with tgb.part(class_name="panel comparison"):
                    tgb.text("## Сравнение режимов", mode="md")
                    tgb.text("{comparison_text}", class_name="muted", id="comparison-summary")
                    tgb.table("{comparison}", columns=COMPARE_COLUMNS, show_all=True)
            with tgb.part(class_name="chart-grid"):
                with tgb.part(class_name="panel"):
                    tgb.text("## Стоимость ошибок", mode="md")
                    tgb.text("Ромб — минимум; контурный круг — выбранный порог.", class_name="muted")
                    tgb.chart(figure="{cost_fig}", height="260px", config=CONFIG)
                with tgb.part(class_name="panel"):
                    tgb.text("## Fβ по порогам", mode="md")
                    tgb.text("Ромб — максимум; контурный круг — выбранный порог.", class_name="muted")
                    tgb.chart(figure="{f_fig}", height="260px", config=CONFIG)
            with tgb.part(class_name="panel examples"):
                tgb.text("## Карточки рядом с порогом", mode="md")
                tgb.selector("{error_label}", lov=ERROR_LABELS, dropdown=True, label="Тип ошибки",
                             id="error-control", width="min(100%, 20rem)", on_change=change_examples)
                tgb.text("{examples_note}", class_name="muted", id="examples-note")
                tgb.table("{examples}", columns=ERROR_COLUMNS, page_size=10, height="430px")
            tgb.text("Учебный лог практики 3.6. Линии показаны на сетке 0,002; оптимумы вычислены по всем оценкам. "
                     "Оценки сохранены заранее; новый инференс не запускается.",
                     class_name="footer muted")
    return page


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--port", type=int, default=5013)
    args = parser.parse_args()
    try:
        log = load(args.data_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    threshold, k, error_label = BASELINE, DEFAULT_COST, ERROR_LABELS[0]
    dataset_caption = (
        f"{integer(log.total)} прогнозов в журнале · {integer(len(log.labeled))} в метриках · "
        f"{integer(log.missing_label + log.excluded_label)} без бинарной разметки"
    )
    globals().update(view_values(log, threshold, k, error_label))
    Gui(page=build_page(), css_file="styles.css").run(
        title="Инференс модели СОК · ИАиВД", host="127.0.0.1", port=args.port,
        dark_mode=False, use_reloader=False, run_browser=False, debug=False,
    )
