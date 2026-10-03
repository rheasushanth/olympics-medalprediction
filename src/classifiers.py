"""Step 10: the five classifiers. Question: will this country win AT LEAST ONE medal?

Each classifier is tuned by 3-fold cross-validation on 1992-2008, then scored
(accuracy) on 2012. The paper reports 0.855 to 0.891, best = Gaussian Naive Bayes.

Run from the project root:   python src/classifiers.py
Other files import:          from classifiers import fit_classifier, classifier_mask
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import FEATURES, NOTEBOOKS, load_features, split_by_year

warnings.filterwarnings("ignore")
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


def _pipe(model):
    # Scale first: GDP is ~1e13 while growth is ~1.
    return Pipeline([("scale", StandardScaler()), ("model", model)])


# name -> (untuned pipeline, search space, "grid" | "random")
def candidates():
    return {
        "LogisticRegression": (_pipe(LogisticRegression(max_iter=5000)),
                               {"model__C": np.logspace(-3, 3, 13)}, "grid"),
        "SVC": (_pipe(SVC()),
                {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1],
                 "model__kernel": ["linear", "rbf"]}, "grid"),
        "GaussianNB": (_pipe(GaussianNB()),
                       {"model__var_smoothing": np.logspace(-12, -1, 12)}, "grid"),
        "MLP": (_pipe(MLPClassifier(max_iter=2000, random_state=0)),
                {"model__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
                 "model__alpha": np.logspace(-5, -1, 5),
                 "model__learning_rate_init": [0.001, 0.01]}, "random"),
        "RandomForest": (_pipe(RandomForestClassifier(random_state=0)),
                         {"model__n_estimators": [100, 200, 400],
                          "model__max_depth": [None, 5, 10],
                          "model__min_samples_leaf": [1, 2, 5]}, "random"),
    }


def fit_classifier(name, train, tuned=True):
    """Fit classifier `name` on `train` to predict Medals > 0 (tuned by cross-validation)."""
    pipe, space, kind = candidates()[name]
    X, y = train[FEATURES], train["Medals"] > 0
    if not tuned:
        return pipe.fit(X, y)
    if kind == "random":
        search = RandomizedSearchCV(pipe, space, n_iter=12, cv=CV, scoring="accuracy", random_state=0)
    else:
        search = GridSearchCV(pipe, space, cv=CV, scoring="accuracy")
    return search.fit(X, y).best_estimator_


def classifier_mask(name, train, target):
    """True for the countries in `target` that classifier `name` predicts will win a medal."""
    return fit_classifier(name, train).predict(target[FEATURES])


if __name__ == "__main__":
    tune, val, _ = split_by_year(load_features())
    y_val = val["Medals"] > 0
    print(f"Tuning on {len(tune)} rows (1992-2008), checking on {len(val)} rows (2012); "
          f"{y_val.sum()} of them really won a medal.")
    print(f"Always guessing 'no medal' would score {1 - y_val.mean():.3f}; "
          f"always guessing 'medal' would score {y_val.mean():.3f}.\n")

    rows = []
    for name in candidates():
        before = fit_classifier(name, tune, tuned=False).predict(val[FEATURES])
        clf = fit_classifier(name, tune)
        pred = clf.predict(val[FEATURES])
        rows.append({
            "Classifier": name,
            "Accuracy before": round((before == y_val).mean(), 3),
            "Accuracy after": round((pred == y_val).mean(), 3),
            "Missed medalists": int((y_val & ~pred).sum()),
            "False alarms": int((~y_val & pred).sum()),
        })
        print("done", name)
    res = pd.DataFrame(rows)
    print("\nClassifier results on 2012:")
    print(res.to_string(index=False))
    res.to_csv(NOTEBOOKS / "classifier_results.csv", index=False)
    print("\nSaved notebooks/classifier_results.csv")
    print("Paper's Table 3 for comparison: LogReg 0.877, SVM 0.884, RandomForest 0.877, "
          "GaussianNB 0.891, MLP 0.855")
