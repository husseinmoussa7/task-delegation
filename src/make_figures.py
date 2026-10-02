"""Regenerate the three figures the manuscript includes.

    python src/make_figures.py

Writes, with the filenames the .tex expects:

    figures/occupation_task_replacement_openai.png    Fig. 2
    figures/subcluster_task_replacement_openai.png    Fig. 3
    figures/best_match_score_distribution_openai.png  Appendix D

Styling follows Visualization_tasks.ipynb. Two differences from that notebook,
both deliberate:

  - It wrote to a hardcoded ~/Downloads path on one particular machine. Output
    now goes to figures/ inside the repository.
  - Its sub-cluster cell used a `grouped_subcluster` variable it never defined;
    the definition lived in a second notebook (task_3_embed_dist_openai.ipynb),
    so the notebook could not run top to bottom. It is defined here.
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import FIGURES, core_tasks, load_scored_tasks

BAR_COLOUR = "0.35"


def figure_2_occupation():
    """Share of Core tasks above the cutoff, by occupation.

    29 bars, not 31: Data Scientists and "Sales Representatives of Services"
    have no Core tasks in O*NET, so they cannot appear. The appendix table
    lists all 31 occupations in the cluster.
    """
    core = core_tasks()
    pivot = (
        core.groupby(["Occupation", "TaskType"])["AI_replaced"]
        .mean()
        .reset_index()
        .pivot(index="Occupation", columns="TaskType", values="AI_replaced")
        .fillna(0)
    )
    # Ascending => largest at the top once drawn with barh. Occupation is a
    # secondary key purely so repeated runs are identical: nine occupations tie
    # (two at 0.21, two at 0.17, two at 0.13, two at 0.08, six at 0.00) and
    # without it their order is whatever the groupby happened to produce. Tie
    # order carries no meaning and may differ from the published figure; the
    # values do not.
    pivot = pivot.sort_values(
        ["Core", "Occupation"], ascending=[True, False], kind="stable"
    )
    labels = pivot.index.tolist()
    vals = pivot["Core"].to_numpy()
    y = np.arange(len(labels))

    fig_h = min(max(0.19 * len(labels) + 4, 6), 16)
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, fig_h))

    bars = ax.barh(y, vals, height=0.26, color=BAR_COLOUR)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", pad=6)

    ax.set_xlim(0, float(vals.max()) + 0.02)
    ax.set_xlabel("Proportion of Tasks with AI Agent Deployment", fontsize=9)
    ax.set_ylabel("Occupation", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)

    for r, v in zip(bars, vals):
        ax.text(
            v,
            r.get_y() + r.get_height() / 2,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )

    # 70% of the width is given to the occupation labels, which are long.
    fig.subplots_adjust(left=0.70, right=0.98, top=0.95, bottom=0.10)
    out = FIGURES / "occupation_task_replacement_openai.png"
    fig.savefig(out, dpi=300, transparent=True)
    plt.close(fig)
    return out, len(labels)


def figure_3_subcluster():
    """Share of Core tasks above the cutoff, by sub-cluster."""
    core = core_tasks()
    grouped = (
        core.groupby("Sub-Cluster")["AI_replaced"]
        .mean()
        .reset_index()
        .sort_values("AI_replaced", ascending=True)
    )
    labels = grouped["Sub-Cluster"].tolist()
    vals = grouped["AI_replaced"].to_numpy()
    y = np.arange(len(labels))

    # Four short bars in a wide, shallow frame. The notebook used figsize=(6, ...)
    # with left=0.70, which left so little axes width that the x-axis label ran
    # off the canvas and the ticks fell at 0.25 intervals rather than 0.2.
    fig_h = max(0.55 * len(labels) + 0.9, 2.4)
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8.2, fig_h))

    ax.barh(y, vals, height=0.42, align="center", color=BAR_COLOUR)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.margins(y=0)

    ax.set_xlim(0, 1.0)
    # Singular "AI Agent" matches the published figure; the notebook said "Agents".
    ax.set_xlabel("Proportion of Core Tasks Fulfilled by AI Agent", fontsize=9)
    ax.set_ylabel("Sub-Cluster", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)

    for yi, v in zip(y, vals):
        ax.text(
            v + 0.01, yi, f"{v:.2f}", va="center", ha="left",
            fontsize=11, fontweight="bold",
        )

    fig.subplots_adjust(left=0.34, right=0.97, top=0.94, bottom=0.26)
    out = FIGURES / "subcluster_task_replacement_openai.png"
    fig.savefig(out, dpi=300, transparent=True)
    plt.close(fig)
    return out, len(labels)


def figure_appendix_distribution():
    """Deployment-score distribution over all 605 marketing tasks.

    All tasks, not only Core — this is the task-level variation underlying the
    occupation and sub-cluster comparisons.
    """
    df = load_scored_tasks()
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["best_match_score"], bins=50, color=BAR_COLOUR)
    ax.set_xlabel("Deployment Score")
    ax.set_ylabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "best_match_score_distribution_openai.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out, len(df)


def main():
    FIGURES.mkdir(exist_ok=True)
    for fn in (figure_2_occupation, figure_3_subcluster, figure_appendix_distribution):
        out, n = fn()
        print(f"  wrote {out.relative_to(out.parents[1])}  ({n} rows)")


if __name__ == "__main__":
    main()
