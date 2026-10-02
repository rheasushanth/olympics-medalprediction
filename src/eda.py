"""Step 8: scatter plot of every feature against medals won (paper's Fig. 2).

Run from the project root:   python src/eda.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FEATURES, NOTEBOOKS, TARGET, load_features

if __name__ == "__main__":
    df = load_features()
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for ax, col in zip(axes.ravel(), FEATURES):
        ax.scatter(df[col], df[TARGET], s=6, alpha=0.5)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Medals won")
    for ax in axes.ravel()[len(FEATURES):]:
        ax.axis("off")
    plt.tight_layout()
    path = NOTEBOOKS / "fig2_scatter.png"
    plt.savefig(path, dpi=130)
    print("Saved", path)

    print("\nCorrelation of each feature with Medals (closer to 1 = moves together):")
    print(df[FEATURES].corrwith(df[TARGET]).sort_values(ascending=False).round(2).to_string())
