"""Run the original Dataset 1 Taipy application."""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from taipy.gui import Gui, State
from taipy.gui import builder as tgb

from analysis import (
    ALL, CLOSED, DEFAULT_DATA, OPEN, SEVERITY, count, day_profile, decimal,
    load_cards, rule_counts, severity_counts, share, subset,
)

BLUE, AMBER, GRAY = "#2855b6", "#a14c0e", "#91a0b8"
INK, GRID = "#18263d", "#e3eaf4"
CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}
TABLE_COLUMNS = ["Карточка", "Создана (UTC)", "Правило", "Критичность", "Состояние", "Вердикт"]


def figure(height=247, bottom=42, left=46):
    fig = go.Figure()
    fig.update_layout(
        height=height, margin=dict(l=left, r=18, t=12, b=bottom),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="system-ui, sans-serif", size=12, color=INK),
        hoverlabel=dict(bgcolor="white", font_color=INK),
        showlegend=False,
    )
    fig.update_xaxes(fixedrange=True, gridcolor=GRID, zeroline=False, automargin=True)
    fig.update_yaxes(fixedrange=True, gridcolor=GRID, zeroline=False, automargin=True, rangemode="tozero")
    return fig


def empty():
    fig = figure()
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text="Нет данных для выбранного среза", x=.5, y=.5, xref="paper",
                       yref="paper", showarrow=False)
    return fig


def trend_chart(df, start, end):
    months = pd.period_range(start, end, freq="M")
    values = df.groupby("month").size().reindex(months, fill_value=0)
    fig = figure()
    fig.add_bar(x=months.to_timestamp(), y=values.to_numpy(), marker_color=BLUE,
                hovertemplate="%{x|%m.%Y}<br>Карточек: %{y:,}<extra></extra>")
    fig.update_xaxes(tickformat="%m.%y", showgrid=False, nticks=7)
    return fig


def week_chart(df, start, end):
    rows = {name: (total, days) for name, (total, days) in day_profile(df, start, end).items() if days}
    fig = figure()
    values = [total / days for total, days in rows.values()]
    fig.add_bar(x=list(rows), y=values, marker_color=BLUE,
                text=[decimal(value, 0) for value in values], textposition="auto",
                customdata=list(rows.values()),
                hovertemplate="%{x}<br>%{y:.1f} карточки/день<br>%{customdata[0]:,} карточек / %{customdata[1]} дней<extra></extra>")
    fig.update_xaxes(showgrid=False)
    return fig


def rules_chart(df):
    top = rule_counts(df).head(10).iloc[::-1]
    fig = figure(300, 32, 40)
    labels = [f"{i}. {name if len(name) <= 22 else name[:19] + '…'}"
              for i, name in zip(range(len(top), 0, -1), top.index)]
    fig.add_bar(x=top.to_numpy(), y=labels, orientation="h", marker_color=BLUE,
                customdata=list(top.index),
                hovertemplate="%{customdata}<br>%{x:,} карточек<extra></extra>")
    fig.update_yaxes(type="category", showgrid=False)
    fig.update_xaxes(rangemode="tozero")
    return fig


def verdict_chart(df):
    levels = severity_counts(df)
    fig = figure(300, 38)
    for key, label, color in [
        ("fp", "Ложное срабатывание", BLUE),
        ("other", "Прочие вердикты", GRAY),
        ("unknown", "Без вердикта", AMBER),
    ]:
        values = levels[key] / levels.total * 100
        fig.add_bar(x=values, y=list(levels.index), orientation="h", name=label, marker_color=color,
                    customdata=list(zip(levels[key], levels.total)),
                    hovertemplate="%{y} · " + label + "<br>%{x:.1f}% (%{customdata[0]:,} из %{customdata[1]:,})<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def dashboard_values(df, dates):
    first, last = (pd.Timestamp(x).tz_localize(None).normalize() for x in dates)
    n = len(df)
    rows = df.head(200)
    table = pd.DataFrame(dict(zip(TABLE_COLUMNS, (
        rows.id, rows.created.dt.strftime("%d.%m.%Y %H:%M"), rows.rule, rows.severity,
        rows.open.map({True: "Открыта", False: "Закрыта"}), rows.verdict,
    ))))
    top = rule_counts(df).head(10)
    return {
        "cards_kpi": count(n),
        "open_kpi": count(int(df.open.sum())),
        "unknown_kpi": share(int(df.unreviewed.sum()), n),
        "fp_kpi": share(int(df.fp.sum()), n),
        "scope": (
            f"{first:%d.%m.%Y} — {last:%d.%m.%Y} · {count(n)} карточек"
            if n else "Нет карточек. Измените фильтры или нажмите «Сбросить»."
        ),
        "rules_note": f"Доля первых 10 правил в срезе: {share(int(top.sum()), n)}.",
        "table_note": f"Показано {count(len(table))} из {count(n)} карточек, начиная с самых новых.",
        "trend": trend_chart(df, first, last) if n else empty(),
        "week": week_chart(df, first, last) if n else empty(),
        "rules": rules_chart(df) if n else empty(),
        "verdicts": verdict_chart(df) if n else empty(),
        "cards_table": table,
    }


def refresh(state: State):
    if not state.dates or len(state.dates) != 2 or any(value is None for value in state.dates):
        return
    df = subset(cards, state.dates, state.severity, state.source, state.status)
    for name, value in dashboard_values(df, state.dates).items():
        state.assign(name, value)


def filters_changed(state: State, _name, _value):
    refresh(state)


def reset(state: State):
    for name, value in {"dates": default_dates.copy(), "severity": ALL, "source": ALL, "status": ALL}.items():
        state.assign(name, value)
    refresh(state)


def metric(label, value, note):
    with tgb.part(class_name="metric"):
        tgb.text(label, class_name="metric-label")
        tgb.text(value, class_name="metric-value")
        tgb.text(note, class_name="muted")


def panel(title, caption, data, height="250px"):
    with tgb.part(class_name="panel"):
        tgb.text(title, mode="md", class_name="panel-title")
        tgb.text(caption, class_name="muted chart-note")
        tgb.chart(figure=data, config=CONFIG, height=height)


def build_page():
    with tgb.Page() as page:
        with tgb.part(class_name="soc-shell"):
            with tgb.part(class_name="hero"):
                tgb.text("ИАиВД / ML-ИНЖИНИРИНГ / ДАТАСЕТ 01", class_name="eyebrow")
                tgb.text("# Карточки СОК", mode="md", class_name="hero-title")
                tgb.text("Как меняется поток карточек, какие правила создают нагрузку и где не хватает вердиктов.",
                         class_name="hero-copy")
            with tgb.part(class_name="filter-box"):
                tgb.text("## Фильтры", mode="md")
                with tgb.part(class_name="filter-grid"):
                    tgb.date_range("{dates}", label_start="Созданы с", label_end="по",
                                   format="dd.MM.yyyy", on_change=filters_changed, class_name="date-control")
                    tgb.selector("{severity}", lov="{severity_options}", dropdown=True, label="Критичность", id="severity-filter",
                                 width="100%", on_change=filters_changed)
                    tgb.selector("{source}", lov="{source_options}", dropdown=True, label="Источник", id="source-filter",
                                 width="100%", on_change=filters_changed)
                    tgb.selector("{status}", lov=[ALL, OPEN, CLOSED], dropdown=True, label="Состояние", id="status-filter",
                                 width="100%", on_change=filters_changed)
                    tgb.button("Сбросить", on_action=reset, class_name="reset-button")
                tgb.text("{scope}", class_name="scope")
            with tgb.part(class_name="metric-grid"):
                metric("Карточки", "{cards_kpi}", "В текущем срезе")
                metric("Открытые", "{open_kpi}", "Без фазы «Закрыт» в выгрузке")
                metric("Без вердикта", "{unknown_kpi}", "Доля от всех карточек среза")
                metric("Ложные срабатывания", "{fp_kpi}", "По вердикту аналитика")
            with tgb.part(class_name="chart-grid"):
                panel("## Поток по месяцам", "Карточки по дате создания; крайние месяцы могут быть неполными.", "{trend}")
                panel("## Будни и выходные", "Среднее на календарный день, включая дни без карточек (UTC).", "{week}")
                panel("## Правила с наибольшим потоком", "{rules_note}", "{rules}", "300px")
                with tgb.part(class_name="panel"):
                    tgb.text("## Вердикты по критичности", mode="md", class_name="panel-title")
                    tgb.text("Доля от всех карточек уровня; размер группы доступен при наведении.", class_name="muted chart-note")
                    tgb.text("**Синий:** ложное срабатывание · **серый:** прочие · **охра:** без вердикта",
                             mode="md", class_name="chart-key")
                    tgb.chart(figure="{verdicts}", config=CONFIG, height="300px")
            with tgb.part(class_name="panel queue"):
                tgb.text("## Карточки среза", mode="md")
                tgb.text("{table_note}", class_name="muted")
                tgb.table("{cards_table}", columns=TABLE_COLUMNS, page_size=10, show_all=False, height="460px")
            tgb.text("Учебная выгрузка 1.5 · Время UTC · Один EntityId = одна карточка, которая может объединять несколько алертов.",
                     class_name="footer muted")
    return page


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Путь к parquet из практики 1.5")
    parser.add_argument("--port", type=int, default=5011)
    args = parser.parse_args()
    try:
        cards = load_cards(args.data)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    default_dates = [cards.day.min().to_pydatetime(), cards.day.max().to_pydatetime()]
    dates = default_dates.copy()
    severity, source, status = ALL, ALL, ALL
    severity_options = [ALL, *[value for value in SEVERITY if value in set(cards.severity)]]
    source_options = [ALL, *cards.source.value_counts().index]
    globals().update(dashboard_values(cards, dates))
    Gui(page=build_page(), css_file="styles.css").run(
        title="Карточки СОК · ИАиВД", host="127.0.0.1", port=args.port, dark_mode=False,
        use_reloader=False, run_browser=False, debug=False,
    )
