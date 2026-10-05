"""Render the redesigned lecture mockup; values are illustrative, not SOC data.

Data source: lecture 2.2, PDF page 27, labeled values in the teacher's example.
The original chart is preserved separately as before.png (PDF page 19).
Run with Python and matplotlib; outputs are written beside this script.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder = Path(__file__).resolve().parent
    with (folder / "mockup_data.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rules = [row["rule"] for row in rows]
    values = [int(row["alerts"]) for row in rows]
    largest = max(values)
    focus = values.index(largest)
    neutral, accent, ink = "#B8C2CC", "#1D5B99", "#25313D"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": ink,
        "axes.labelcolor": ink,
        "xtick.color": ink,
        "ytick.color": "#5F6B77",
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(12, 7), facecolor="white")
    ax = fig.add_axes((0.10, 0.20, 0.86, 0.55))
    colors = [accent if index == focus else neutral for index in range(len(values))]
    bars = ax.bar(rules, values, width=0.59, color=colors, edgecolor="none", zorder=2)
    for index, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.2,
            str(value),
            ha="center",
            va="bottom",
            fontsize=14,
            color=accent if index == focus else ink,
            fontweight="bold" if index == focus else "normal",
        )

    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Количество алертов", labelpad=14)
    ax.set_xlabel("Правило SIEM", labelpad=14)
    ax.tick_params(axis="both", length=0, pad=10)
    ax.grid(False)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#8C98A5")
    ax.spines["bottom"].set_linewidth(0.8)
    labels = ax.get_xticklabels()
    labels[focus].set_color(accent)
    labels[focus].set_fontweight("bold")

    fig.text(0.10, 0.91, "Алерты по правилам SIEM за сутки", fontsize=23, fontweight="bold")
    fig.text(0.10, 0.85, f"Максимум у {rules[focus]}: {largest} алертов за сутки.", fontsize=15)
    fig.text(
        0.10, 0.075,
        "Данные условные. Синим выделено наибольшее количество алертов; критичность цветом не кодируется.",
        fontsize=10, color="#5F6B77",
    )
    fig.text(0.10, 0.035, "Источник: лекция 2.2, страница 27. Значения и порядок правил сохранены.", fontsize=10, color="#5F6B77")
    fig.savefig(folder / "after.png", dpi=180, facecolor="white")
    fig.savefig(folder / "after.svg", facecolor="white")
    plt.close(fig)
    print("Created after.png and after.svg")
    print("Values:", dict(zip(rules, values)))


if __name__ == "__main__":
    main()
