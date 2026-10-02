"""Shared helpers: load features.csv and split it by year (paper's Table 2).

Note: our cleaned data starts in 1992, because the 1988 rows have no
"medals at the previous Games" value and were dropped in Step 6.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "features.csv"
NOTEBOOKS = ROOT / "notebooks"

# Model inputs, in the order of the paper's Table 1. NOC is NOT a feature.
FEATURES = [
    "Year", "GDP", "GDP_per_capita", "GDP_growth", "GDP_share",
    "Pop", "Pop_share", "Pop_growth", "Area",
    "Athletes", "Athletes_share", "Medals_last", "Medals_last_share",
]
TARGET = "Medals"


def load_features():
    return pd.read_csv(FEATURES_CSV)


def split_by_year(df):
    """Return (tune, val, test): tune = up to 2008, val = 2012, test = 2016."""
    tune = df[df["Year"] <= 2008]
    val = df[df["Year"] == 2012]
    test = df[df["Year"] == 2016]
    return tune, val, test


def xy(df):
    return df[FEATURES], df[TARGET]
