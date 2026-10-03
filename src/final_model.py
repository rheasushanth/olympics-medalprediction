"""Steps 13 and 14: choose the best classifier + regressor pair, then run the final 2016 test.

Step 13: every tuned classifier is paired with every tuned regressor. Each pair is scored on
         2012 (trained on 1992-2008) with the paper's combined loss = Loss1 + 0.25 * Loss2.
         The pair with the lowest combined loss is chosen.
Step 14: the chosen pair is retrained on 1992-2012 and predicts 2016. The classifier says who
         medals; the regressor says how many; everyone else gets 0.

Run from the project root:   python src/final_model.py
Needs (from Person 2):       common.py, evaluate.py, regressors.py, baseline.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from baseline import make_baseline
from classifiers import candidates as clf_candidates
from classifiers import classifier_mask
from common import FEATURES, NOTEBOOKS, load_features, split_by_year
from evaluate import evaluate
from regressors import candidates as reg_candidates
from regressors import fit_tuned, score, two_step_predict

warnings.filterwarnings("ignore")
PAPER_PAIR = ("GaussianNB", "Ridge")  # the paper's final choice


def step13(tune, val):
    """Score every classifier + regressor pair on 2012."""
    regs = {r: fit_tuned(r, tune) for r in reg_candidates()}
    rows = []
    for c in clf_candidates():
        mask = classifier_mask(c, tune, val)
        for r, model in regs.items():
            s = score(model, val, mask)
            rows.append({"Classifier": c, "Regressor": r, "RMSE": s["rmse"],
                         "Loss1": s["loss1"], "Loss2": s["loss2"], "Combined": s["combined"]})
    return pd.DataFrame(rows).sort_values("Combined").reset_index(drop=True)


if __name__ == "__main__":
    df = load_features()
    tune, val, test = split_by_year(df)

    # ---- Step 13 -----------------------------------------------------------------
    res = step13(tune, val)
    res.round(3).to_csv(NOTEBOOKS / "pair_results_2012.csv", index=False)
    print("Step 13: best 10 classifier + regressor pairs on 2012 (lower Combined is better):")
    print(res.head(10).round(3).to_string(index=False))
    paper = res[(res.Classifier == PAPER_PAIR[0]) & (res.Regressor == PAPER_PAIR[1])].iloc[0]
    print(f"\nThe paper's pair {PAPER_PAIR[0]} + {PAPER_PAIR[1]}: RMSE {paper.RMSE:.3f}, "
          f"Combined {paper.Combined:.2f}")
    best = res.iloc[0]
    print(f"Chosen pair: {best.Classifier} + {best.Regressor} (Combined {best.Combined:.2f})")

    # ---- Step 14 -----------------------------------------------------------------
    train = df[df["Year"] <= 2012]
    print("\nStep 14: retraining on 1992-2012 and predicting 2016 ...")

    def run(c, r):
        mask = classifier_mask(c, train, test)
        model = fit_tuned(r, train)
        return two_step_predict(model, test, mask), mask

    pred, mask = run(best.Classifier, best.Regressor)
    final = evaluate(test["Medals"], pred, test["Year"])
    pred_paper, _ = run(*PAPER_PAIR)
    paper_final = evaluate(test["Medals"], pred_paper, test["Year"])
    base = make_baseline().fit(train[FEATURES], train["Medals"])
    base_final = evaluate(test["Medals"], base.predict(test[FEATURES]), test["Year"])

    out = pd.DataFrame([
        {"Model": "Baseline linear regression", "RMSE 2016": base_final["rmse"], "Combined": base_final["combined"]},
        {"Model": f"Paper's pair ({PAPER_PAIR[0]} + {PAPER_PAIR[1]})", "RMSE 2016": paper_final["rmse"], "Combined": paper_final["combined"]},
        {"Model": f"Our chosen pair ({best.Classifier} + {best.Regressor})", "RMSE 2016": final["rmse"], "Combined": final["combined"]},
    ]).round(3)
    print("\nFinal test on 2016 (paper: baseline 3.25, final 2.26):")
    print(out.to_string(index=False))
    out.to_csv(NOTEBOOKS / "final_results_2016.csv", index=False)

    print(f"\nClassifier let {mask.sum()} of {len(test)} countries through "
          f"({(test.Medals > 0).sum()} really won medals).")
    top = test.assign(Predicted=pred.round(0).astype(int)).sort_values("Medals", ascending=False).head(5)
    print("\nTop 5 countries, 2016 (actual vs predicted):")
    print(top[["NOC", "Medals", "Predicted"]].to_string(index=False))

    plt.figure(figsize=(6, 6))
    plt.scatter(test["Medals"], pred, s=14)
    lim = max(test["Medals"].max(), pred.max())
    plt.plot([0, lim], [0, lim], "r-")
    plt.xlabel("Actual medals (2016)")
    plt.ylabel("Predicted medals")
    plt.title(f"Final model: {best.Classifier} + {best.Regressor}, 2016")
    plt.tight_layout()
    plt.savefig(NOTEBOOKS / "fig5_final.png", dpi=130)
    print("\nSaved notebooks/fig5_final.png")
