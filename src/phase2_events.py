"""Phase 2, final: new clues from the entry lists, and a fair comparison with the paper.

NEW CLUES (all known BEFORE the Games, just like the paper's "Athletes" clue):
  Events          how many different events the country enters
  TeamEvents      how many of those are team events (a team of 15 can win only 1 medal)
  Slots           medal chances: 1 per team event, up to 3 athletes per individual event
  Events_share,   the same as % of all countries
  Slots_share
  NextHost        1 if the country hosts the NEXT Games (hosts build up their teams early)
  Eff_last        medals per slot at the previous Games (how efficient the team was)
  ExpMedals       Eff_last x this year's Slots (a simple "expected medals" guess)

  Also tried, kept in the table, but NOT chosen:
  RetMedalists, RetMedals, RetMedals2, KeptEvents, MedalsLastSameEvents
    (how many of this year's athletes already won medals, and in which events)

FAIR RULE: the model and clue set are chosen ONLY from 2004 and 2008 (each predicted using
earlier Games). 2012 and 2016 are then reported once, as test years.

WHY 2012 MATTERS: the paper's Table 4 (labelled 2016) lists USA 103, China 89, UK 65,
Germany 44, Japan 38. Those are the 2012 counts; the real 2016 counts were 121, 70, 67, 42, 41.
So the paper's final 2.26 most likely describes 2012 data, and we compare on both years.

Run from the project root:   python src/phase2_events.py      (takes a few minutes)
Needs:                       data/raw/athlete_events.csv, data/processed/features.csv
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FEATURES, NOTEBOOKS, ROOT, load_features
from evaluate import evaluate
from phase2 import HOSTS, add_host, make_model

warnings.filterwarnings("ignore")
HOSTS_ALL = {**HOSTS, 2020: "JPN"}  # 2020 is only needed for "NextHost" in 2016
CHOOSE_YEARS = [2004, 2008]
TEST_YEARS = [2012, 2016]

EVENT_CLUES = ["Host", "NextHost", "Events", "Events_share", "TeamEvents", "Slots", "Slots_share",
               "Eff_last", "ExpMedals"]
RETURNING_CLUES = ["RetMedalists", "RetMedals", "RetMedals2", "KeptEvents", "MedalsLastSameEvents"]
CLUE_SETS = {
    "Paper's clues": FEATURES,
    "Paper's clues + host": FEATURES + ["Host"],
    "+ entry-list clues": FEATURES + EVENT_CLUES,
    "+ entry-list + returning medalists": FEATURES + EVENT_CLUES + RETURNING_CLUES,
}
MODELS = {
    "Ridge": ["Ridge"],
    "GradBoost": ["GradBoost"],
    "RandomForest": ["RandomForest"],
    "Ensemble GB+RF": ["GradBoost", "RandomForest"],
    "Ensemble GB+RF+Ridge": ["GradBoost", "RandomForest", "Ridge"],
}

# Paper's numbers (Table 3 = validation on 2012; Section 6 = its "final" result)
PAPER_2012 = {"Paper: Random Forest (Table 3)": 2.20, "Paper: Ridge (Table 3)": 2.27,
              "Paper: final model (reported as 2016)": 2.26}
PAPER_BASELINE = 3.25


def entry_list_clues(features):
    """Add the new clues to features.csv using the raw athlete list."""
    a = pd.read_csv(ROOT / "data" / "raw" / "athlete_events.csv")
    a = a[(a["Season"] == "Summer") & (a["Year"] >= 1980)]
    games = sorted(a["Year"].unique())
    prev = dict(zip(games[1:], games[:-1]))
    medals = a.dropna(subset=["Medal"])

    per_event = a.groupby(["Year", "NOC", "Event"])["ID"].nunique().reset_index()
    per_event["team"] = per_event["ID"] > 2
    per_event["slots"] = np.where(per_event["team"], 1, np.minimum(per_event["ID"], 3))
    agg = per_event.groupby(["Year", "NOC"]).agg(Events=("Event", "nunique"),
                                                 TeamEvents=("team", "sum"),
                                                 Slots=("slots", "sum")).reset_index()

    rows = []
    for y in sorted(features["Year"].unique()):
        py = prev[y]
        ppy = prev.get(py)
        roster = a[a["Year"] == y][["NOC", "ID", "Event"]]
        ids = roster[["NOC", "ID"]].drop_duplicates()
        m1 = medals[medals["Year"] == py][["ID", "Event", "Medal", "NOC"]].rename(columns={"NOC": "NOC_then"})
        m2 = medals[medals["Year"].isin([py, ppy])][["ID", "Year", "Event", "Medal", "NOC"]].rename(
            columns={"NOC": "NOC_then"})
        r1 = ids.merge(m1, on="ID")
        r1u = r1.drop_duplicates(["NOC", "Event", "Medal", "NOC_then"])
        r2 = ids.merge(m2, on="ID").drop_duplicates(["NOC", "Year", "Event", "Medal", "NOC_then"])
        ev_now = roster[["NOC", "Event"]].drop_duplicates()
        ev_last = medals[medals["Year"] == py][["NOC", "Event"]].drop_duplicates()
        kept = ev_now.merge(ev_last, on=["NOC", "Event"])
        same = (medals[medals["Year"] == py].drop_duplicates(["NOC", "Event", "Medal"])[["NOC", "Event"]]
                .merge(ev_now, on=["NOC", "Event"]))
        g = pd.DataFrame({"NOC": ids["NOC"].unique(), "Year": y})
        for name, s in [("RetMedalists", r1.groupby("NOC")["ID"].nunique()),
                        ("RetMedals", r1u.groupby("NOC").size()),
                        ("RetMedals2", r2.groupby("NOC").size()),
                        ("KeptEvents", kept.groupby("NOC").size()),
                        ("MedalsLastSameEvents", same.groupby("NOC").size())]:
            g = g.merge(s.rename(name).reset_index(), on="NOC", how="left")
        rows.append(g)
    returning = pd.concat(rows).fillna(0)

    df = add_host(features).merge(agg, on=["Year", "NOC"], how="left")
    df = df.merge(returning, on=["Year", "NOC"], how="left").fillna({c: 0 for c in RETURNING_CLUES})
    for c in ["Events", "Slots"]:
        df[c + "_share"] = 100 * df[c] / df.groupby("Year")[c].transform("sum")
    df["NextHost"] = [int(HOSTS_ALL.get(y + 4) == n) for y, n in zip(df["Year"], df["NOC"])]
    slots = agg.set_index(["NOC", "Year"])["Slots"]
    df["Slots_last"] = [slots.get((n, prev.get(y)), np.nan) for n, y in zip(df["NOC"], df["Year"])]
    df["Eff_last"] = (df["Medals_last"] / df["Slots_last"]).fillna(df["Medals_last"] / df["Slots"].clip(lower=1))
    df["ExpMedals"] = df["Eff_last"] * df["Slots"]
    return df


def predict(train, test, cols, parts):
    preds = [make_model(m).fit(train[cols], train["Medals"]).predict(test[cols]) for m in parts]
    return np.clip(np.mean(preds, axis=0), 0, None)


def score_everything(df):
    rows = []
    for clue_name, cols in CLUE_SETS.items():
        for y in CHOOSE_YEARS + TEST_YEARS:
            train, test = df[df["Year"] < y], df[df["Year"] == y]
            for model_name, parts in MODELS.items():
                rmse = evaluate(test["Medals"], predict(train, test, cols, parts), test["Year"])["rmse"]
                rows.append({"Clues": clue_name, "Model": model_name, "Year": y, "RMSE": rmse})
    table = pd.DataFrame(rows).pivot_table(index=["Clues", "Model"], columns="Year", values="RMSE")
    table["Choose on (mean 2004+2008)"] = table[CHOOSE_YEARS].mean(axis=1)
    return table.sort_values("Choose on (mean 2004+2008)")


def baseline(df, year):
    train, test = df[df["Year"] < year], df[df["Year"] == year]
    return evaluate(test["Medals"], predict(train, test, FEATURES, ["Linear"]), test["Year"])["rmse"]


def draw(ours12, ours16, base12, base16, path):
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor("#fcfcfb")
    panels = [
        ("2012: the year the paper's numbers describe", [
            ("Paper: Random Forest (Table 3)", 2.20, blue), ("Paper: final model*", 2.26, blue),
            ("Paper: Ridge (Table 3)", 2.27, blue), ("Ours: final model", ours12, orange),
            ("Ours: baseline", base12, orange)]),
        ("2016: the true test year", [
            ("Ours: final model", ours16, orange), ("Ours: baseline", base16, orange),
            ("Paper: baseline", PAPER_BASELINE, blue),
            ("Paper's chosen pair, run on our data", 3.478, orange)]),
    ]
    for ax, (title, rows) in zip(axes, panels):
        rows = rows[::-1]
        for i, (name, v, c) in enumerate(rows):
            ax.barh(i, v, color=c, height=0.6)
            ax.text(v + 0.04, i, f"{v:.2f}", va="center", fontsize=11, color=ink)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=ink)
        ax.set_xlim(0, 4.0)
        ax.set_xlabel("Error in medals (lower is better)", color=muted)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=ink)
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=blue, label="Paper"), Patch(color=orange, label="Ours")],
               loc="upper right", ncol=2, frameon=False, fontsize=11)
    fig.text(0.01, 0.01, "* The paper reports 2.26 as its 2016 result, but its 2016 table holds the 2012 medal "
             "counts, so we place it with 2012.", fontsize=9, color=muted)
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    df = entry_list_clues(load_features())
    table = score_everything(df)
    table.round(3).to_csv(NOTEBOOKS / "phase2_events_results.csv")
    print("Every clue set x model. Chosen ONLY by the mean of 2004 and 2008 (lower is better).\n")
    print(table.round(3).to_string())

    chosen_clues, chosen_model = table.index[0]
    ours12, ours16 = table.iloc[0][2012], table.iloc[0][2016]
    base12, base16 = baseline(df, 2012), baseline(df, 2016)
    print(f"\nChosen: {chosen_model} with '{chosen_clues}'")

    print("\n=== 2012 (trained on 1992-2008). The year the paper's numbers describe ===")
    for k, v in PAPER_2012.items():
        print(f"  {k:42s} {v:.2f}")
    print(f"  {'Ours: baseline (straight line)':42s} {base12:.2f}")
    print(f"  {'Ours: final model':42s} {ours12:.2f}")

    print("\n=== 2016 (trained on 1992-2012). The true test year ===")
    print(f"  {'Paper: baseline':42s} {PAPER_BASELINE:.2f}")
    print(f"  {'Paper: chosen pair, run on our data':42s} 3.48")
    print(f"  {'Ours: baseline (straight line)':42s} {base16:.2f}")
    print(f"  {'Ours: final model':42s} {ours16:.2f}")

    best = MODELS[chosen_model]
    train, test = df[df["Year"] < 2016], df[df["Year"] == 2016]
    pred = predict(train, test, CLUE_SETS[chosen_clues], best)
    out = test[["NOC", "Medals"]].assign(Predicted=pred.round(0)).sort_values("Medals", ascending=False)
    print("\nTop 5 countries, 2016 (actual vs predicted):")
    print(out.head(5).to_string(index=False))

    # ---- The paper's apparent protocol: train through 2012, then score the 2012 rows ----
    # The paper's "2016" table holds the 2012 counts and its model hits USA 103 and Germany 44
    # exactly, which suggests it scored rows it had already trained on. Scoring this way is NOT a
    # fair test of prediction; we show it only to compare like with like with the paper's 2.26.
    from classifiers import classifier_mask
    from regressors import fit_tuned, score

    base = load_features()
    tr_p, te_p = base[base["Year"] <= 2012], base[base["Year"] == 2012]
    pair = score(fit_tuned("Ridge", tr_p), te_p, classifier_mask("GaussianNB", tr_p, te_p))["rmse"]
    tr_o, te_o = df[df["Year"] <= 2012], df[df["Year"] == 2012]
    ours_same = evaluate(te_o["Medals"], predict(tr_o, te_o, CLUE_SETS[chosen_clues], best),
                         te_o["Year"])["rmse"]
    print("\n=== Paper's apparent protocol (train through 2012, score the 2012 rows; NOT a fair test) ===")
    print(f"  {'Paper: final model as reported':42s} 2.26")
    print(f"  {'Paper: chosen pair, our data, same protocol':42s} {pair:.2f}")
    print(f"  {'Ours: final model, same protocol':42s} {ours_same:.2f}")
    pd.DataFrame({"Model": ["Paper: final model as reported", "Paper: chosen pair, our data",
                            "Ours: final model"], "RMSE": [2.26, round(pair, 3), round(ours_same, 3)]}
                 ).to_csv(NOTEBOOKS / "paper_protocol_2012.csv", index=False)

    draw(ours12, ours16, base12, base16, NOTEBOOKS / "fig7_final_vs_paper.png")
    print("\nSaved notebooks/phase2_events_results.csv and notebooks/fig7_final_vs_paper.png")
