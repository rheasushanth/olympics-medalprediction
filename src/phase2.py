"""Phase 2: try to beat the baseline on the true 2016 test.

Phase 1 showed the two-step model has trouble with the big countries (USA, China, Russia).
Here we try three ideas, each on its own and together:

  1. Target: predict the CHANGE from last Games (Medals - Medals_last) instead of the raw count,
     then add Medals_last back. This lets a model predict above anything it saw in training.
  2. Host: add a 0/1 clue "is this country hosting these Games?" (hosts win extra medals).
  3. Model: gradient boosting (HistGradientBoosting-style, via GradientBoostingRegressor).

Fair test: every setting is chosen on 2012 (trained on 1992-2008). Only then is the chosen
setting retrained on 1992-2012 and scored once on 2016.

Run from the project root:   python src/phase2.py
Needs:                       common.py, evaluate.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, NOTEBOOKS, load_features
from evaluate import evaluate

warnings.filterwarnings("ignore")

# Host country of each Summer Games in our data (NOC codes as in features.csv)
HOSTS = {1992: "ESP", 1996: "USA", 2000: "AUS", 2004: "GRE", 2008: "CHN", 2012: "GBR", 2016: "BRA"}


def add_host(df):
    df = df.copy()
    df["Host"] = (df["NOC"] == df["Year"].map(HOSTS)).astype(int)
    return df


def make_model(name):
    if name == "Linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0)
    if name == "GradBoost":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                         subsample=0.8, random_state=0)
    raise ValueError(name)


def run(train, test, model_name, target, use_host):
    """Train on `train`, predict `test`. Returns predicted medals (never below 0)."""
    cols = FEATURES + (["Host"] if use_host else [])
    y_tr = train["Medals"] - train["Medals_last"] if target == "change" else train["Medals"]
    model = make_model(model_name).fit(train[cols], y_tr)
    pred = model.predict(test[cols])
    if target == "change":
        pred = pred + test["Medals_last"].to_numpy()
    return np.clip(pred, 0, None)


def score_all(train, test):
    rows = []
    for model_name in ["Linear", "Ridge", "RandomForest", "GradBoost"]:
        for target in ["raw", "change"]:
            for use_host in [False, True]:
                pred = run(train, test, model_name, target, use_host)
                s = evaluate(test["Medals"], pred, test["Year"])
                rows.append({"Model": model_name, "Target": target, "Host": use_host,
                             "RMSE": s["rmse"], "Combined": s["combined"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = add_host(load_features())
    tune, val, test = df[df.Year <= 2008], df[df.Year == 2012], df[df.Year == 2016]
    trainval = df[df.Year <= 2012]

    # Step A: choose on 2012 (train 1992-2008)
    r12 = score_all(tune, val).rename(columns={"RMSE": "RMSE_2012", "Combined": "Combined_2012"})
    # Step B: the same settings retrained on 1992-2012 and scored on 2016
    r16 = score_all(trainval, test).rename(columns={"RMSE": "RMSE_2016", "Combined": "Combined_2016"})
    res = r12.merge(r16, on=["Model", "Target", "Host"]).sort_values("Combined_2012")
    res.round(3).to_csv(NOTEBOOKS / "phase2_results.csv", index=False)

    print("Phase 2: all settings, sorted by 2012 Combined (lower is better).")
    print("Pick on 2012; 2016 is only the final exam.\n")
    print(res.round(3).to_string(index=False))

    best = res.iloc[0]
    base = res[(res.Model == "Linear") & (res.Target == "raw") & (~res.Host)].iloc[0]
    print(f"\nChosen on 2012: {best.Model}, target={best.Target}, host={best.Host}")
    print(f"  2016 RMSE: {best.RMSE_2016:.3f}   (baseline linear: {base.RMSE_2016:.3f}, "
          f"Phase 1 two-step: 3.269, paper: 2.26)")

    # ---- Fairer test: 2016 alone is one noisy year, so repeat for 2004, 2008, 2012, 2016 ----
    # Each year is predicted using only the Games BEFORE it. Ensembles average several models.
    SPECS = {
        "Baseline linear": [("Linear", "raw", False)],
        "Ridge + host": [("Ridge", "raw", True)],
        "GradBoost + host": [("GradBoost", "raw", True)],
        "RandomForest + host": [("RandomForest", "raw", True)],
        "Ensemble GB + Ridge + host": [("GradBoost", "raw", True), ("Ridge", "raw", True)],
        "Ensemble GB + RF + Ridge + host": [("GradBoost", "raw", True), ("RandomForest", "raw", True),
                                            ("Ridge", "raw", True)],
    }
    years = [2004, 2008, 2012, 2016]
    rows = []
    for name, spec in SPECS.items():
        rm = []
        for y in years:
            tr, te = df[df.Year < y], df[df.Year == y]
            pred = np.mean([run(tr, te, m, t, h) for m, t, h in spec], axis=0)
            rm.append(evaluate(te["Medals"], pred, te["Year"])["rmse"])
        rows.append({"Method": name, **{str(y): v for y, v in zip(years, rm)},
                     "Mean 2008-2016": float(np.mean(rm[1:]))})
    roll = pd.DataFrame(rows)
    roll.round(3).to_csv(NOTEBOOKS / "phase2_rolling.csv", index=False)
    print("\nRMSE for each Games, trained only on earlier Games (lower is better):")
    print(roll.round(3).to_string(index=False))

    # Top countries for the best ensemble on 2016
    spec = SPECS["Ensemble GB + RF + Ridge + host"]
    pred = np.mean([run(trainval, test, m, t, h) for m, t, h in spec], axis=0)
    out = test[["NOC", "Medals"]].assign(Predicted=pred.round(0)).sort_values("Medals", ascending=False)
    print("\nTop 5 countries, 2016, best ensemble (actual vs predicted):")
    print(out.head(5).to_string(index=False))
