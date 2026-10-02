"""Step 11: the six regressors (how many medals?), tuned with cross-validation.

Following the paper:
  * regressors are TRAINED only on countries that won at least one medal;
  * a classifier (Gaussian Naive Bayes, the paper's best) decides which 2012
    countries the regressors see; every other country is predicted as 0;
  * we score the combined two-step prediction on ALL 2012 countries.

Run from the project root:   python src/regressors.py          (paper's Gaussian NB classifier)
                             python src/regressors.py logreg   (Logistic Regression classifier)
Person 1 imports:            from regressors import get_ridge, get_mask
"""
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, PoissonRegressor, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import FEATURES, NOTEBOOKS, load_features, split_by_year
from evaluate import evaluate

warnings.filterwarnings("ignore")
CV = KFold(n_splits=3, shuffle=True, random_state=0)


def _pipe(model):
    # Scaling matters: GDP is ~1e13 while growth is ~1.
    return Pipeline([("scale", StandardScaler()), ("model", model)])


# name -> (untuned pipeline, search space or None, "grid" | "random")
def candidates():
    return {
        "Linear": (_pipe(LinearRegression()), None, "grid"),
        "Ridge": (_pipe(Ridge()), {"model__alpha": np.logspace(-3, 3, 13)}, "grid"),
        "Lasso": (_pipe(Lasso(max_iter=50000)), {"model__alpha": np.logspace(-3, 1, 9)}, "grid"),
        "Poisson": (_pipe(PoissonRegressor(max_iter=5000)), {"model__alpha": np.logspace(-4, 1, 6)}, "grid"),
        "SVR": (_pipe(SVR(kernel="linear")),
                {"model__C": [0.1, 1, 10, 100], "model__epsilon": [0.1, 0.5, 1.0]}, "grid"),
        "RandomForest": (_pipe(RandomForestRegressor(random_state=0)),
                         {"model__n_estimators": [100, 200, 400],
                          "model__max_depth": [None, 5, 10, 20],
                          "model__min_samples_leaf": [1, 2, 5]}, "random"),
    }


def medalists_only(df):
    return df[df["Medals"] > 0]


def gnb_mask(train, target):
    """True for countries in `target` that Gaussian Naive Bayes predicts will win a medal."""
    clf = GaussianNB().fit(train[FEATURES], train["Medals"] > 0)
    return clf.predict(target[FEATURES])


def logreg_mask(train, target):
    """Same idea with scaled Logistic Regression (on our data it beats Gaussian NB, see Step 10)."""
    clf = _pipe(LogisticRegression(max_iter=5000)).fit(train[FEATURES], train["Medals"] > 0)
    return clf.predict(target[FEATURES])


def get_mask(kind, train, target):
    """kind is "gnb" (the paper's choice) or "logreg"."""
    return {"gnb": gnb_mask, "logreg": logreg_mask}[kind](train, target)


def fit_tuned(name, train):
    """Tune regressor `name` by cross-validation on the medal-winning rows of `train`."""
    pipe, space, kind = candidates()[name]
    rows = medalists_only(train)
    X, y = rows[FEATURES], rows["Medals"]
    if space is None:
        return pipe.fit(X, y)
    if kind == "random":
        search = RandomizedSearchCV(pipe, space, n_iter=15, cv=CV, random_state=0,
                                    scoring="neg_mean_squared_error")
    else:
        search = GridSearchCV(pipe, space, cv=CV, scoring="neg_mean_squared_error")
    return search.fit(X, y).best_estimator_


def fit_untuned(name, train):
    rows = medalists_only(train)
    return candidates()[name][0].fit(rows[FEATURES], rows["Medals"])


def get_ridge(train):
    """Tuned Ridge, fitted on the medal-winning rows of `train`. Person 1 uses this in Steps 13-14."""
    return fit_tuned("Ridge", train)


def two_step_predict(model, target, mask):
    """Predict 0 where the classifier says no medal, the regressor's number otherwise."""
    pred = np.zeros(len(target))
    if mask.any():
        pred[mask] = np.clip(model.predict(target.loc[mask, FEATURES]), 0, None)
    return pred


def score(model, target, mask):
    pred = two_step_predict(model, target, mask)
    return evaluate(target["Medals"], pred, target["Year"])


if __name__ == "__main__":
    tune, val, _ = split_by_year(load_features())
    kind = sys.argv[1] if len(sys.argv) > 1 else "gnb"
    mask = get_mask(kind, tune, val)
    print(f"Classifier used to pick medal countries: {kind}")
    print(f"2012: classifier lets {mask.sum()} of {len(val)} countries through "
          f"({(val['Medals'] > 0).sum()} really won medals)\n")

    rows = []
    for name in candidates():
        before = score(fit_untuned(name, tune), val, mask)
        after = score(fit_tuned(name, tune), val, mask)
        rows.append({"Regressor": name, "RMSE before": before["rmse"], "RMSE after": after["rmse"],
                     "Combined after": after["combined"]})
        print(f"done {name}")
    res = pd.DataFrame(rows).round(3)
    print("\nRegressor results on 2012 (two-step, all countries):")
    print(res.to_string(index=False))
    res.to_csv(NOTEBOOKS / f"regressor_results_{kind}.csv", index=False)

    x = np.arange(len(res))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.2, res["RMSE before"], 0.4, label="Before tuning")
    plt.bar(x + 0.2, res["RMSE after"], 0.4, label="After tuning")
    plt.xticks(x, res["Regressor"], rotation=20)
    plt.ylabel("RMSE on 2012")
    plt.title("Regressors before and after tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(NOTEBOOKS / f"fig4_regressors_{kind}.png", dpi=130)
    print(f"\nSaved fig4_regressors_{kind}.png and regressor_results_{kind}.csv in notebooks/")
