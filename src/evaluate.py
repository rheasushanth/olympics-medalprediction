"""The error measures used in the paper.

Loss1 = average squared error over all countries.
Loss2 = average squared error over each Games' top-10 medal winners only.
The paper chooses models by  Loss1 + 0.25 * Loss2  ("combined").
Its "average std dev" looks like the square root of Loss1 (RMSE); we report
RMSE and say so in the write-up.
"""
import numpy as np
import pandas as pd


def evaluate(y_true, y_pred, years=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if years is None:
        years = np.zeros(len(y_true))
    sq = (y_true - y_pred) ** 2
    loss1 = sq.mean()
    d = pd.DataFrame({"y": y_true, "sq": sq, "year": np.asarray(years)})
    top10 = d.sort_values("y", ascending=False).groupby("year").head(10)
    loss2 = top10["sq"].mean()
    return {
        "rmse": float(np.sqrt(loss1)),
        "loss1": float(loss1),
        "loss2": float(loss2),
        "combined": float(loss1 + 0.25 * loss2),
    }


if __name__ == "__main__":
    # tiny self-test: perfect predictions give zero error
    print(evaluate([0, 3, 10, 50], [0, 3, 10, 50]))
    print(evaluate([0, 3, 10, 50], [1, 3, 8, 40]))
