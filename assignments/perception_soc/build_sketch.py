"""Render a SOC overview sketch from six explicitly synthetic cards.

These values and severity categories are illustrative, not dataset findings.
Run with Python and Matplotlib; PNG/SVG outputs are written beside the script.
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def main():
    folder = Path(__file__).resolve().parent
    with (folder / "mock_cards.csv").open(encoding="utf-8", newline="") as stream:
        cards = list(csv.DictReader(stream))
    counts = Counter(card["Критичность"] for card in cards)
    focus = [card for card in cards if card["Критичность"] == "Высокая" and card["Вердикт"] == "Не указан"]
    total = len({card["EntityId"] for card in cards})
    ink, muted, line = "#243345", "#526276", "#DCE3EB"
    accent, pale = "#9B4A00", "#FFF1DE"
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(14, 10), dpi=160)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 1400), ylim=(1000, 0))
    ax.axis("off")
    fig.set_facecolor("#F5F7FA")

    def text(x, y, content, size=12, color=ink, weight="normal", **kwargs):
        ax.text(x, y, content, fontsize=size, color=color, fontweight=weight, va="center", **kwargs)

    def panel(x, y, width, height, color="white", edge=line, radius=10):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=color, edgecolor=edge, linewidth=1))

    text(50, 49, "Обзор карточек SOC", 25, weight="bold")
    text(50, 87, "ЭСКИЗ · Все карточки и значения условные, не строки учебного датасета", 11, muted)
    panel(50, 111, 1300, 46)
    text(70, 135, "Период создания: 06.10.2026, 00:00–23:59 · условный день", 12)
    text(1330, 135, "Единица подсчёта: карточка", 11, muted, ha="right")

    panel(50, 181, 300, 213)
    text(73, 211, "Всего карточек", 13, weight="bold")
    text(73, 238, "за выбранный период", 11, muted)
    text(73, 283, str(total), 37, weight="bold")
    text(73, 325, "Уникальные EntityId", 12, muted)
    text(73, 360, "Не число исходных алертов", 11, muted)

    panel(370, 181, 480, 213)
    text(393, 211, "Количество карточек по критичности", 13, weight="bold")
    start, scale = 521, 76
    ax.plot([start, start], [238, 340], color="#9AA8B8", linewidth=1)
    for index, level in enumerate(["Высокая", "Средняя", "Низкая"]):
        y = 250 + index * 37
        text(393, y, level, 12)
        width = counts[level] * scale
        ax.add_patch(Rectangle((start, y-10), width, 20, facecolor="#8395AA", edgecolor="none"))
        text(start+width+12, y, str(counts[level]), 12, weight="bold")
    for value in range(4):
        text(start+value*scale, 363, str(value), 10, muted, ha="center")
    text(635, 382, "карточки", 9, muted, ha="center")

    panel(870, 181, 480, 213, color=pale, edge="#D2A16D")
    text(893, 211, "Приоритет проверки", 16, accent, "bold")
    text(893, 261, f"{len(focus)} карточка", 25, accent, "bold")
    text(893, 310, "Высокая критичность + «Не указан»", 12, accent)
    text(893, 351, "Условие внимания, не диагноз атаки", 11, muted)

    panel(50, 420, 1300, 437)
    text(73, 448, "Карточки за выбранный период", 15, weight="bold")
    text(1326, 448, "Порядок: время создания", 10, muted, ha="right")
    ax.add_patch(Rectangle((70, 474), 690, 31, facecolor="#EEF2F7", edgecolor="none"))
    ax.add_patch(Rectangle((774, 474), 552, 31, facecolor="#EEF2F7", edgecolor="none"))
    text(87, 489, "КОНТЕКСТ СОБЫТИЯ", 10, muted, "bold")
    text(791, 489, "ОЦЕНКА И РАЗБОР", 10, muted, "bold")
    for x, label in [(87,"EntityId"),(265,"Создан"),(483,"Наименование / правило"),(791,"Критичность"),(992,"Вердикт аналитика")]:
        text(x, 527, label, 11, weight="bold")
    ax.plot([70,1326], [550,550], color=line, linewidth=1)
    verdicts = {
        "Не указан": ("НУ", "Не указан"),
        "False Positive": ("FP", "False Positive"),
        "Решен": ("РШ", "Решен"),
        "Дубликат": ("ДУБ", "Дубликат"),
        "Закрыт из родительской атаки": ("РОД", "Из родительской атаки"),
    }
    for index, card in enumerate(cards):
        y = 575 + index * 45
        is_focus = card in focus
        if is_focus:
            ax.add_patch(Rectangle((70, y-20), 1256, 40, facecolor=pale, edgecolor="none"))
            ax.add_patch(Rectangle((70, y-20), 4, 40, facecolor=accent, edgecolor="none"))
        text(87, y, card["EntityId"], 12, weight="bold" if is_focus else "normal")
        text(265, y, card["Создан"][11:], 12)
        text(483, y, card["Наименование"], 12)
        text(791, y, card["Критичность"], 12, accent if is_focus else ink, "bold" if is_focus else "normal")
        code, label = verdicts[card["Вердикт"]]
        panel(984, y-15, 46, 30, color="#EEF2F7", edge="#EEF2F7", radius=5)
        text(1007, y, code, 9, muted, "bold", ha="center")
        text(1040, y, label, 10)
        ax.plot([70,1326], [y+23,y+23], color=line, linewidth=0.6)
    text(87, 836, "Акцент строки: высокая критичность и ещё не указан вердикт", 11, accent)

    text(50, 889, "НУ — Не указан · FP — False Positive · РШ — Решен · ДУБ — Дубликат", 11, muted)
    text(50, 916, "РОД — Закрыт из родительской атаки · Коды меток не являются числовым полем verdict", 10, muted)
    text(50, 953, "Критичность: условная шкала макета · Вердикт: запись аналитика, не модельный скор", 10, muted)
    text(50, 978, "Источник структуры: справочник iavd_incidents_raw · Значения: синтетический пример из mock_cards.csv", 10, muted)

    fig.savefig(folder / "sketch.png", facecolor=fig.get_facecolor())
    fig.savefig(folder / "sketch.svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Rendered sketch: {total} cards; severity counts {dict(counts)}; attention condition {len(focus)}.")


if __name__ == "__main__":
    main()
