"""Phase 2, paper-matched check: the same models with Russia left out.

WHY: the paper's top-5 table for 2012 lists USA 103, China 89, UK 65, Germany 44, Japan 38.
Russia won 82 medals in 2012 and should be 3rd, but it is not there, so the paper's data most
likely had no Russia rows. In our data Russia alone is about 20% of the 2012 error. To compare
like with like, we drop Russia (RUS) from every year, as the paper's data appears to have done.

Everything else is identical to phase2_events.py: same clues, same models, and the model is
still chosen ONLY from 2004 and 2008. Nothing earlier is changed; this only adds new output files.

Run from the project root:   python src/phase2_no_russia.py      (takes a few minutes)
"""
import warnings

import pandas as pd

from common import FEATURES, NOTEBOOKS, load_features
from evaluate import evaluate
from phase2_events import CLUE_SETS, MODELS, PAPER_2012, entry_list_clues, predict, score_everything

warnings.filterwarnings("ignore")


def baseline(df, year):
    train, test = df[df["Year"] < year], df[df["Year"] == year]
    return evaluate(test["Medals"], predict(train, test, FEATURES, ["Linear"]), test["Year"])["rmse"]


if __name__ == "__main__":
    full = entry_list_clues(load_features())
    df = full[full["NOC"] != "RUS"].copy()
    print(f"Rows: {len(full)} with Russia, {len(df)} without Russia\n")

    table = score_everything(df)
    table.round(3).to_csv(NOTEBOOKS / "phase2_no_russia_results.csv")
    print("WITHOUT RUSSIA. Every clue set x model, chosen ONLY by the mean of 2004 and 2008:\n")
    print(table.round(3).to_string())

    chosen_clues, chosen_model = table.index[0]
    ours12, ours16 = table.iloc[0][2012], table.iloc[0][2016]
    print(f"\nChosen: {chosen_model} with '{chosen_clues}'")

    # The same chosen model WITH Russia, for comparison
    parts, cols = MODELS[chosen_model], CLUE_SETS[chosen_clues]
    with_rus = {}
    for y in (2012, 2016):
        tr, te = full[full["Year"] < y], full[full["Year"] == y]
        with_rus[y] = evaluate(te["Medals"], predict(tr, te, cols, parts), te["Year"])["rmse"]

    print("\n=== 2012 (trained on 1992-2008), the year the paper's numbers describe ===")
    for k, v in PAPER_2012.items():
        print(f"  {k:44s} {v:.2f}")
    print(f"  {'Ours: baseline, without Russia':44s} {baseline(df, 2012):.2f}")
    print(f"  {'Ours: final model, WITH Russia':44s} {with_rus[2012]:.2f}")
    print(f"  {'Ours: final model, WITHOUT Russia':44s} {ours12:.2f}")

    print("\n=== 2016 (trained on 1992-2012), the true test year ===")
    print(f"  {'Paper: baseline':44s} 3.25")
    print(f"  {'Ours: baseline, without Russia':44s} {baseline(df, 2016):.2f}")
    print(f"  {'Ours: final model, WITH Russia':44s} {with_rus[2016]:.2f}")
    print(f"  {'Ours: final model, WITHOUT Russia':44s} {ours16:.2f}")

    pd.DataFrame({
        "Comparison": ["2012 paper best (RF, Table 3)", "2012 paper final (reported as 2016)",
                       "2012 ours with Russia", "2012 ours without Russia",
                       "2016 paper baseline", "2016 ours with Russia", "2016 ours without Russia"],
        "RMSE": [2.20, 2.26, round(with_rus[2012], 3), round(ours12, 3),
                 3.25, round(with_rus[2016], 3), round(ours16, 3)],
    }).to_csv(NOTEBOOKS / "no_russia_summary.csv", index=False)
    print("\nSaved notebooks/phase2_no_russia_results.csv and notebooks/no_russia_summary.csv")
