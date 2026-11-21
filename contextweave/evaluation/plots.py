"""Report figures (matplotlib, static PNG for the paper).

Palette: validated categorical slots 1-3 (blue, orange, aqua) plus neutral grey;
text stays in ink colours, the grid is recessive, and series are direct-labelled
so identity never relies on colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, GRID, SURFACE = "#9a9893", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def quality_vs_cost(sweep: list[dict], points: dict[str, dict], title: str, path: Path) -> Path:
    """sweep: [{cost, ndcg}] for the controller over tau x budget (grey cloud + Pareto front);
    points: {label: {cost, ndcg}} baselines / systems (labelled markers)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    xs, ys = [p["cost"] for p in sweep], [p["ndcg"] for p in sweep]
    ax.scatter(xs, ys, s=10, color=GREY, alpha=0.45, linewidths=0, label="controller settings (τ × budget)")
    front, best = [], -1.0
    for p in sorted(sweep, key=lambda p: (p["cost"], -p["ndcg"])):
        if p["ndcg"] > best:
            front.append(p)
            best = p["ndcg"]
    ax.plot([p["cost"] for p in front], [p["ndcg"] for p in front], color=BLUE, linewidth=2,
            label="Pareto front")
    colours = {"baseline": INK2, "hybrid": ORANGE, "ours": AQUA}
    ys_all = [p["ndcg"] for p in points.values()]
    span = (max(ys_all) - min(ys_all)) or 1.0
    placed: list[tuple[float, float]] = []  # (log-cost, ndcg) of labels already placed
    import math
    for label, p in sorted(points.items(), key=lambda kv: -kv[1]["ndcg"]):
        c = colours.get(p.get("kind", "baseline"), INK2)
        ax.scatter([p["cost"]], [p["ndcg"]], s=46, color=c, edgecolors=SURFACE, linewidths=1.5, zorder=5)
        lx = math.log10(max(p["cost"], 0.1))
        # crude collision avoidance: step the label down while it is near an earlier label
        dy = 4
        while any(abs(lx - x) < 0.12 and abs((p["ndcg"] - y) / span * 300 + (dy - yo)) < 10 for x, y, yo in
                  [(a, b, c_) for a, b, c_ in placed]):
            dy -= 10
        placed.append((lx, p["ndcg"], dy))
        ax.annotate(label, (p["cost"], p["ndcg"]), textcoords="offset points", xytext=(7, dy), fontsize=7.5,
                    color=INK, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6) if dy < -6 else None)
    ax.set_xscale("symlog", linthresh=1.0)
    _style(ax, title, "mean cost per query (cost units, 1 = one full BM25 query; symlog)", "NDCG@10")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def by_type_bars(data: dict[str, dict[str, float]], types: list[str], title: str, path: Path) -> Path:
    """data: {system_label: {type: ndcg}} — up to 3 systems as grouped bars."""
    systems = list(data)[:3]
    colours = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    width = 0.8 / len(systems)
    for i, s in enumerate(systems):
        xs = [j + (i - (len(systems) - 1) / 2) * width for j in range(len(types))]
        vals = [data[s].get(t, 0.0) for t in types]
        ax.bar(xs, vals, width=width - 0.03, color=colours[i], label=s, edgecolor=SURFACE, linewidth=1)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(types)), types)
    _style(ax, title, "query type", "NDCG@10")
    ax.legend(frameon=False, fontsize=7.5, ncol=len(systems), loc="upper center", bbox_to_anchor=(0.5, -0.18),
              labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def latency_bars(rows: list[tuple[str, float, float]], title: str, path: Path) -> Path:
    """rows: [(label, median_ms, p95_ms)] — horizontal bars (median) with a p95 tick."""
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(rows) + 1.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    labels = [r[0] for r in rows][::-1]
    med = [r[1] for r in rows][::-1]
    p95 = [r[2] for r in rows][::-1]
    ax.barh(labels, med, color=BLUE, height=0.6, label="median")
    ax.scatter(p95, labels, marker="|", s=120, color=INK, label="95th percentile", zorder=5)
    for y, (m, p) in enumerate(zip(med, p95)):
        ax.text(max(m, p) * 1.15 + 0.05, y, f"{m:.1f} / {p:.1f} ms", va="center", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(right=max(p95) * 6)
    _style(ax, title, "latency per query (ms, log scale, warm runs)", "")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
