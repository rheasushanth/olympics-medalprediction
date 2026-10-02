"""Step 12: weighted regression (paper's Eq. 4).

Each medal-winning training row gets weight  w = exp(-(y - y0)^2 / (2 * tau^2)),
with y0 = 50 and tau = 1 in the paper. Rows far from y0 get weight ~0, so the
model concentrates on the top-scoring countries.

Run from the project root:   python src/weighted.py [gnb|logreg]
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, load_features, split_by_year
from regressors import get_mask, get_ridge, medalists_only, score

warnings.filterwarnings("ignore")
Y0 = 50


def weights(y, tau, y0=Y0):
    return np.exp(-((np.asarray(y) - y0) ** 2) / (2 * tau ** 2))


def fit_weighted(model, train, tau):
    rows = medalists_only(train)
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    return pipe.fit(rows[FEATURES], rows["Medals"], model__sample_weight=weights(rows["Medals"], tau))


if __name__ == "__main__":
    tune, val, _ = split_by_year(load_features())
    kind = sys.argv[1] if len(sys.argv) > 1 else "gnb"
    mask = get_mask(kind, tune, val)
    print("Classifier used:", kind)
    rows = medalists_only(tune)

    base = score(get_ridge(tune), val, mask)
    print(f"Unweighted tuned Ridge on 2012: RMSE {base['rmse']:.3f}, combined {base['combined']:.2f}\n")

    out = []
    for tau in (1, 5, 10, 20, 40):
        w = weights(rows["Medals"], tau)
        for name, model in (("Linear", LinearRegression()), ("Ridge", Ridge(alpha=1.0)), ("Lasso", Lasso(alpha=0.1, max_iter=50000))):
            r = score(fit_weighted(model, tune, tau), val, mask)
            out.append({"tau": tau, "model": name, "effective rows": round(w.sum(), 1),
                        "RMSE": round(r["rmse"], 3), "combined": round(r["combined"], 2)})
    print(pd.DataFrame(out).to_string(index=False))
    print("\nLook at 'effective rows': with tau = 1 almost no training row has weight, "
          "so the paper's setting leaves very little data to learn from.")
