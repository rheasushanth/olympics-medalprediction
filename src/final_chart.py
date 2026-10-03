"""Draw the final 'paper vs ours' chart (fig8) from the saved result files.

Needs (run these first):  python src/phase2_events.py      -> notebooks/paper_protocol_2012.csv
                          python src/phase2_no_russia.py   -> notebooks/no_russia_summary.csv
Run from the project root: python src/final_chart.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from common import FEATURES, NOTEBOOKS, load_features
from evaluate import evaluate
from phase2_events import predict

BLUE, ORANGE = "#2a78d6", "#eb6834"  # paper, ours
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def our_baseline_2016():
    df = load_features()
    tr, te = df[df["Year"] < 2016], df[df["Year"] == 2016]
    return evaluate(te["Medals"], predict(tr, te, FEATURES, ["Linear"]), te["Year"])["rmse"]


if __name__ == "__main__":
    nr = pd.read_csv(NOTEBOOKS / "no_russia_summary.csv").set_index("Comparison")["RMSE"]
    pp = pd.read_csv(NOTEBOOKS / "paper_protocol_2012.csv").set_index("Model")["RMSE"]

    panels = [
        ("2012, same countries as the paper",
         "(Russia left out, as in the paper's table)", [
             ("Paper: Random Forest (best)", 2.20, BLUE),
             ("Paper: final model", 2.26, BLUE),
             ("Paper: Ridge", 2.27, BLUE),
             ("Ours: final model", nr["2012 ours without Russia"], ORANGE)]),
        ("2016, the true test year", "(all countries)", [
             ("Paper: baseline", 3.25, BLUE),
             ("Paper's chosen pair, on our data", 3.48, ORANGE),
             ("Ours: baseline", our_baseline_2016(), ORANGE),
             ("Ours: final model", nr["2016 ours with Russia"], ORANGE)]),
        ("Paper's own scoring method", "(train through 2012, score 2012)", [
             ("Paper: final model", 2.26, BLUE),
             ("Paper's chosen pair, on our data", pp["Paper: chosen pair, our data"], ORANGE),
             ("Ours: final model", pp["Ours: final model"], ORANGE)]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    fig.patch.set_facecolor(BG)
    for ax, (title, sub, rows) in zip(axes, panels):
        rows = rows[::-1]
        for i, (name, v, c) in enumerate(rows):
            ours_final = name == "Ours: final model"
            ax.barh(i, v, color=c, height=0.6, edgecolor=INK if ours_final else "none",
                    linewidth=1.6 if ours_final else 0)
            ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=12, color=INK,
                    fontweight="bold" if ours_final else "normal")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK)
        ax.set_xlim(0, 4.0)
        ax.set_xlabel("Error in medals (lower is better)", color=MUTED)
        ax.set_title(f"{title}\n", loc="left", fontsize=12.5, fontweight="bold", color=INK)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10, color=MUTED)
        ax.set_facecolor(BG)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)

    fig.legend(handles=[Patch(color=BLUE, label="Paper (its reported numbers)"),
                        Patch(color=ORANGE, label="Ours (run on our data)")],
               loc="upper right", ncol=2, frameon=False, fontsize=11)
    fig.text(0.01, 0.015,
             "The paper's '2016' table holds the 2012 medal counts and leaves out Russia (82 medals in 2012), "
             "so its 2.26 is compared with 2012. The third panel is not a fair test of prediction; it only "
             "scores both models the way the paper did.", fontsize=9, color=MUTED, wrap=True)
    plt.tight_layout(rect=(0, 0.05, 1, 0.93))
    out = NOTEBOOKS / "fig8_final_scoreboard.png"
    plt.savefig(out, dpi=160)
    print("Saved", out)
