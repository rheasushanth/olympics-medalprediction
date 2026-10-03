"""EXTRA data step (Phase 2): put the big Soviet-era teams and the 1988 Games back.

Why: the original features.csv starts in 1992 and has no rows for the Soviet Union (1988) or the
1992 Unified Team, because (a) 1988 has no "previous Games" in our data and (b) the World Bank has
no figures for these teams. They were huge medal winners (131 and 112 medals), and the biggest
medal count in the original training data is only about 110. Our test year has the USA at 121.

What this file changes, compared with build_features.py:
  1. Reads the 1980 and 1984 Games too, only to fill in "medals last time" for the 1988 rows.
     If a team skipped 1984 (the boycott: USSR, East Germany, Cuba, ...), it uses its 1980 medals.
  2. Gives the Soviet Union (1988) and the Unified Team (1992) economy numbers by adding up their
     republics from the World Bank file (GDP scaled up if some republics have no GDP figure).
     Our own assumption: the paper does not say how it handled these teams.
  3. The Unified Team's predecessor is the Soviet Union, so its "medals last time" is 131.

The original features.csv is NOT touched. This writes data/processed/features_extended.csv.

Run from the project root:   python src/extended_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import FEATURES, PREDECESSORS, TARGET
from merge_wdi import NOC_TO_ISO, SERIES, load_wdi

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_CSV = ROOT / "data" / "processed" / "features_extended.csv"
FIRST_LAG_YEAR = 1980  # only used to look up "medals last time"
FIRST_ROW_YEAR = 1988  # rows we model start here (World Bank data starts in 1988)

REPUBLICS_EUN = ["RUS", "UKR", "BLR", "KAZ", "UZB", "GEO", "ARM", "AZE", "MDA", "KGZ", "TJK", "TKM"]
REPUBLICS_URS = REPUBLICS_EUN + ["EST", "LVA", "LTU"]  # the Baltic states competed alone in 1992


def medal_table(path=RAW / "athlete_events.csv"):
    """Like medals.build_medal_table, but from 1980 so we can look back from 1988."""
    df = pd.read_csv(path)
    df = df[(df["Season"] == "Summer") & (df["Year"] >= FIRST_LAG_YEAR)]
    athletes = df.groupby(["Year", "NOC"])["ID"].nunique().rename("Athletes").reset_index()
    medalled = df.dropna(subset=["Medal"])
    medals = (medalled.drop_duplicates(subset=["Year", "NOC", "Event", "Medal"])
              .groupby(["Year", "NOC"]).size().rename("Medals").reset_index())
    out = athletes.merge(medals, on=["Year", "NOC"], how="left")
    out["Medals"] = out["Medals"].fillna(0).astype(int)
    return out


def soviet_economy(wdi, year, republics):
    """Add up the republics' World Bank numbers for one year (a made-up 'country')."""
    w = wdi[(wdi.ISO.isin(republics)) & (wdi.Year == year)]
    pop = w["Pop"].sum()
    have_gdp = w.dropna(subset=["GDP"])
    gdp = have_gdp["GDP"].sum() * (pop / have_gdp["Pop"].sum())  # scale up for republics w/o GDP
    area = wdi[(wdi.ISO.isin(republics)) & (wdi.Year == 1992)]["Area"].sum()  # land does not change
    rus = wdi[(wdi.ISO == "RUS") & (wdi.Year == year)].iloc[0]

    def growth(col):  # Russia's growth if available, else the average of the republics
        v = rus[col]
        return v if pd.notna(v) else w[col].mean()

    return {"GDP": gdp, "Pop": pop, "Area": area, "GDP_per_capita": gdp / pop,
            "GDP_growth": growth("GDP_growth"), "Pop_growth": growth("Pop_growth")}


def build_extended():
    wdi = load_wdi()
    medals = medal_table()
    medals["ISO"] = medals["NOC"].map(lambda c: NOC_TO_ISO.get(c, c))
    df = medals.merge(wdi, on=["ISO", "Year"], how="left")

    # Fill in the Soviet Union (1988) and the Unified Team (1992)
    for noc, year, reps in [("URS", 1988, REPUBLICS_URS), ("EUN", 1992, REPUBLICS_EUN)]:
        eco = soviet_economy(wdi, year, reps)
        idx = df[(df.NOC == noc) & (df.Year == year)].index
        for col, val in eco.items():
            df.loc[idx, col] = val

    # --- shares of the world, as in build_features.py (all teams count in the totals) ---
    world = wdi.query("ISO == 'WLD'").set_index("Year")
    df["GDP_share"] = 100 * df["GDP"] / df["Year"].map(world["GDP"])
    df["Pop_share"] = 100 * df["Pop"] / df["Year"].map(world["Pop"])
    df["Athletes_share"] = 100 * df["Athletes"] / df.groupby("Year")["Athletes"].transform("sum")

    # --- medals at the previous Games ---
    predecessors = dict(PREDECESSORS)
    predecessors["EUN"] = ["URS"]  # the Unified Team's "last time" is the Soviet Union's 1988
    years = sorted(df["Year"].unique())
    prev_year = dict(zip(years[1:], years[:-1]))
    lookup = df.set_index(["Year", "NOC"])["Medals"].to_dict()
    total_by_year = df.groupby("Year")["Medals"].sum().to_dict()

    def last_year_for(noc, year):
        py = prev_year.get(year)
        if py is None:
            return None
        # 1988 only: a team that skipped 1984 (boycott) uses its 1980 medals instead
        if year == 1988 and (py, noc) not in lookup and (1980, noc) in lookup:
            return 1980
        return py

    def medals_last(row):
        py = last_year_for(row["NOC"], row["Year"])
        if py is None:
            return np.nan
        total = lookup.get((py, row["NOC"]), 0)
        for old in predecessors.get(row["NOC"], []):
            total += lookup.get((py, old), 0)
        return total

    df["Medals_last"] = df.apply(medals_last, axis=1)
    df["Last_year_used"] = [last_year_for(n, y) for n, y in zip(df.NOC, df.Year)]
    df["Medals_last_share"] = 100 * df["Medals_last"] / df["Last_year_used"].map(total_by_year)
    return df


if __name__ == "__main__":
    df = build_extended()
    df = df[df.Year >= FIRST_ROW_YEAR]
    before = len(df)
    clean = df.dropna(subset=FEATURES + [TARGET])
    print(f"Rows from 1988 on: {before}, after dropping rows with gaps: {len(clean)}")
    print("Rows per year:", clean.groupby("Year").size().to_dict())
    print("Biggest medal counts now in the data:",
          clean.sort_values("Medals", ascending=False).head(6)[["Year", "NOC", "Medals"]]
          .to_string(index=False).replace("\n", " | "))
    out = clean[["Year", "NOC"] + [f for f in FEATURES if f != "Year"] + [TARGET]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("Saved", OUT_CSV, out.shape)
    print("\nSoviet rows (check they look sensible):")
    cols = ["Year", "NOC", "GDP", "Pop", "Area", "Athletes", "Medals_last", "Medals"]
    print(out[out.NOC.isin(["URS", "EUN"])][cols].to_string(index=False))
