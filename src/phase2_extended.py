"""Does putting the Soviet-era rows and the 1988 Games back lower the error?

Same models and the same test Games (2004, 2008, 2012, 2016) for both tables; the only difference
is the training data. Each Games is predicted using only earlier Games.

Run (after python src/extended_features.py):   python src/phase2_extended.py
"""
import warnings

import numpy as np
import pandas as pd

from common import NOTEBOOKS, ROOT, load_features
from evaluate import evaluate
from phase2 import add_host, run

warnings.filterwarnings("ignore")

SPECS = {
    "Baseline linear": [("Linear", "raw", False)],
    "GradBoost + host": [("GradBoost", "raw", True)],
    "RandomForest + host": [("RandomForest", "raw", True)],
    "Ensemble GB + RF + Ridge + host": [("GradBoost", "raw", True), ("RandomForest", "raw", True),
                                        ("Ridge", "raw", True)],
}
YEARS = [2004, 2008, 2012, 2016]


def rolling(df):
    rows = {}
    for name, spec in SPECS.items():
        rm = []
        for y in YEARS:
            tr, te = df[df.Year < y], df[df.Year == y]
            pred = np.mean([run(tr, te, m, t, h) for m, t, h in spec], axis=0)
            rm.append(evaluate(te["Medals"], pred, te["Year"])["rmse"])
        rows[name] = rm + [float(np.mean(rm[1:]))]
    return pd.DataFrame(rows, index=[str(y) for y in YEARS] + ["Mean 2008-2016"]).T


if __name__ == "__main__":
    orig = add_host(load_features())
    ext = add_host(pd.read_csv(ROOT / "data" / "processed" / "features_extended.csv"))
    a, b = rolling(orig), rolling(ext)
    print("RMSE (lower is better)\n")
    print("ORIGINAL data (1992 onwards, 1300 rows):")
    print(a.round(3).to_string())
    print("\nEXTENDED data (adds 1988 and the Soviet-era rows, 1437 rows):")
    print(b.round(3).to_string())
    print("\nChange (extended minus original; negative = better):")
    print((b - a).round(3).to_string())
    pd.concat({"original": a, "extended": b}, axis=1).round(3).to_csv(NOTEBOOKS / "phase2_extended.csv")

    # Top countries in 2016 with the extended data and the best ensemble
    spec = SPECS["Ensemble GB + RF + Ridge + host"]
    tr, te = ext[ext.Year < 2016], ext[ext.Year == 2016]
    pred = np.mean([run(tr, te, m, t, h) for m, t, h in spec], axis=0)
    out = te[["NOC", "Medals"]].assign(Predicted=pred.round(0)).sort_values("Medals", ascending=False)
    print("\nTop 5 countries, 2016, extended data, best ensemble:")
    print(out.head(5).to_string(index=False))
