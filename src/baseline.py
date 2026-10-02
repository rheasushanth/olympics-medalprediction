"""Step 9: baseline = plain linear regression on ALL countries (no classifier).

Trains on 1992-2012 and predicts 2016 (paper's Fig. 3). The paper's baseline
error was 3.25. This is the number our two-step model has to beat.

Run from the project root:   python src/baseline.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import NOTEBOOKS, load_features, split_by_year, xy
from evaluate import evaluate


def make_baseline():
    return make_pipeline(StandardScaler(), LinearRegression())


if __name__ == "__main__":
    df = load_features()
    tune, val, test = split_by_year(df)

    # Check on 2012 (train on the tuning years only)
    model = make_baseline().fit(*xy(tune))
    r_val = evaluate(val["Medals"], model.predict(val[xy(val)[0].columns]), val["Year"])
    print("Baseline, trained 1992-2008, tested on 2012:", {k: round(v, 3) for k, v in r_val.items()})

    # Final baseline: train 1992-2012, test on 2016
    train = df[df["Year"] <= 2012]
    model = make_baseline().fit(*xy(train))
    pred = model.predict(xy(test)[0])
    r_test = evaluate(test["Medals"], pred, test["Year"])
    print("Baseline, trained 1992-2012, tested on 2016:", {k: round(v, 3) for k, v in r_test.items()})

    plt.figure(figsize=(6, 6))
    plt.scatter(test["Medals"], pred, s=14)
    top = max(test["Medals"].max(), pred.max())
    plt.plot([0, top], [0, top], "r-")
    plt.xlabel("Actual medals (2016)")
    plt.ylabel("Predicted medals")
    plt.title("Baseline linear regression, 2016")
    plt.tight_layout()
    path = NOTEBOOKS / "fig3_baseline.png"
    plt.savefig(path, dpi=130)
    print("Saved", path)

    print("\nTop 5 countries, 2016 (actual vs predicted):")
    t = test.assign(Predicted=pred.round(0)).sort_values("Medals", ascending=False).head(5)
    print(t[["NOC", "Medals", "Predicted"]].to_string(index=False))
