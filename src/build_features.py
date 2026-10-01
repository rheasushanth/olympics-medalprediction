"""Steps 5 and 6: build the paper's features, remove rows with missing values,
draw the null heatmap (paper's Fig. 1) and save data/processed/features.csv.

Run from the project root:
    python src/build_features.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from merge_wdi import SERIES, build_merged, load_wdi

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "processed" / "features.csv"
OUT_FIG = ROOT / "notebooks" / "fig1_null_heatmap.png"

# Teams that were replaced by a successor team. When we look up "medals at the
# previous Games" for the successor, we add the predecessor's medals.
# (Our own choice: the paper does not say how it handled this.)
PREDECESSORS = {
    "GER": ["FRG", "GDR"],  # Germany reunified after 1988
    "RUS": ["EUN"],         # Unified Team 1992 -> Russia
    "CZE": ["TCH"],         # Czechoslovakia -> Czech Republic
    "SRB": ["SCG"],         # Serbia and Montenegro -> Serbia
}

# Columns used as inputs to the models, in the order of the paper's Table 1.
FEATURES = [
    "Year", "GDP", "GDP_per_capita", "GDP_growth", "GDP_share",
    "Pop", "Pop_share", "Pop_growth", "Area",
    "Athletes", "Athletes_share", "Medals_last", "Medals_last_share",
]
TARGET = "Medals"


def add_features(df):
    df = df.copy()

    # --- % of the world (the paper's "normalized" columns) -----------------
    world = load_wdi().query("ISO == 'WLD'").set_index("Year")
    df["GDP_share"] = 100 * df["GDP"] / df["Year"].map(world["GDP"])
    df["Pop_share"] = 100 * df["Pop"] / df["Year"].map(world["Pop"])
    df["Athletes_share"] = 100 * df["Athletes"] / df.groupby("Year")["Athletes"].transform("sum")

    # --- medals at the previous Games --------------------------------------
    years = sorted(df["Year"].unique())
    prev_year = dict(zip(years[1:], years[:-1]))  # 1992 -> 1988, ...
    lookup = df.set_index(["Year", "NOC"])["Medals"].to_dict()
    total_by_year = df.groupby("Year")["Medals"].sum().to_dict()

    def medals_last(row):
        py = prev_year.get(row["Year"])
        if py is None:  # 1988 has no earlier Games in our data (paper uses 1988 onwards)
            return float("nan")
        noc = row["NOC"]
        total = lookup.get((py, noc), 0)  # 0 if the country skipped the previous Games
        for old in PREDECESSORS.get(noc, []):
            total += lookup.get((py, old), 0)
        return total

    df["Medals_last"] = df.apply(medals_last, axis=1)
    df["Medals_last_share"] = 100 * df["Medals_last"] / df["Year"].map(prev_year).map(total_by_year)
    return df


def draw_null_heatmap(df, path):
    cols = FEATURES + [TARGET]
    plt.figure(figsize=(9, 6))
    sns.heatmap(df[cols].isnull(), cbar=False, yticklabels=False)
    plt.title("Missing values (light = missing), before removing rows")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    df = add_features(build_merged())
    draw_null_heatmap(df, OUT_FIG)
    print("Saved", OUT_FIG)

    before = len(df)
    clean = df.dropna(subset=FEATURES + [TARGET])
    print(f"\nRows before dropping nulls: {before}, after: {len(clean)}")
    print("Rows per year after cleaning:")
    print(clean.groupby("Year").size().to_string())
    print("Medals kept:", int(clean[TARGET].sum()), "of", int(df[TARGET].sum()))

    # NOC is kept only so we can read results by country. Do NOT train on it.
    out = clean[["Year", "NOC"] + [f for f in FEATURES if f != "Year"] + [TARGET]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("Saved", OUT_CSV, out.shape)

    print("\nShares per year should add up to about 100 (or less, since some rows were dropped):")
    print(clean.groupby("Year")[["Athletes_share"]].sum().round(1).T.to_string())
    print("\nUSA 2016 row:")
    print(out[(out.NOC == "USA") & (out.Year == 2016)].T.to_string(header=False))
    print("\nGermany 1992 row (medals_last should include East and West Germany):")
    print(out[(out.NOC == "GER") & (out.Year == 1992)].T.to_string(header=False))
