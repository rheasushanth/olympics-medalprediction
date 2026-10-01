"""Step 3: count medals and athletes per country (NOC) per Summer Games, 1988 onwards.

Run from the project root:
    python src/medals.py
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
FIRST_YEAR = 1988  # paper: skip 1980/84 boycotts and poorer older data


def build_medal_table(path=RAW / "athlete_events.csv"):
    """Return one row per (Year, NOC) with columns: Year, NOC, Athletes, Medals."""
    df = pd.read_csv(path)
    df = df[(df["Season"] == "Summer") & (df["Year"] >= FIRST_YEAR)]

    # Athletes: unique athlete IDs per country per Games.
    athletes = (
        df.groupby(["Year", "NOC"])["ID"].nunique().rename("Athletes").reset_index()
    )

    # Medals: ONE medal per event per country (a team of 15 = 1 medal).
    medalled = df.dropna(subset=["Medal"])
    medals = (
        medalled.drop_duplicates(subset=["Year", "NOC", "Event", "Medal"])
        .groupby(["Year", "NOC"])
        .size()
        .rename("Medals")
        .reset_index()
    )

    # Left join keeps countries that sent athletes but won nothing; fill their medals with 0.
    out = athletes.merge(medals, on=["Year", "NOC"], how="left")
    out["Medals"] = out["Medals"].fillna(0).astype(int)
    return out


if __name__ == "__main__":
    table = build_medal_table()
    print(table.head(10))
    print("rows:", table.shape[0], "| years:", sorted(table["Year"].unique()))
    print("countries with 0 medals:", (table["Medals"] == 0).sum())
    print("\nSanity check, the paper's 2016 top 5 (actual medals):")
    top = table[table["Year"] == 2016].sort_values("Medals", ascending=False).head(5)
    print(top.to_string(index=False))
    print("\nPaper's Table 4 for comparison: USA 103, China 89, UK 65, Germany 44, Japan 38")
