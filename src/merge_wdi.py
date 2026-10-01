"""Step 4: attach World Bank indicators (GDP, population, land area) to the medal table.

The Olympics data uses Olympic (NOC) country codes; the World Bank uses ISO codes.
They differ for many countries, so NOC_TO_ISO below translates the ones that differ.
A value of None means "no World Bank data exists for this team" (the row is dropped).

Run from the project root:
    python src/merge_wdi.py
"""
from pathlib import Path

import pandas as pd

from medals import build_medal_table

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# NOC code -> World Bank (ISO) code, only where they differ. Anything not listed
# here is assumed to use the same code in both (checked against country names).
NOC_TO_ISO = {
    "ALG": "DZA", "ANG": "AGO", "ANT": "ATG", "ARU": "ABW", "ASA": "ASM",
    "BAH": "BHS", "BAN": "BGD", "BAR": "BRB", "BER": "BMU", "BHU": "BTN",
    "BIZ": "BLZ", "BOT": "BWA", "BRU": "BRN", "BUL": "BGR", "BUR": "BFA",
    "CAM": "KHM", "CAY": "CYM", "CGO": "COG", "CHA": "TCD", "CRC": "CRI",
    "CRO": "HRV", "DEN": "DNK", "ESA": "SLV", "FIJ": "FJI", "GAM": "GMB",
    "GBS": "GNB", "GEQ": "GNQ", "GER": "DEU", "GRE": "GRC", "GRN": "GRD",
    "GUA": "GTM", "GUI": "GIN", "HAI": "HTI", "HON": "HND", "INA": "IDN",
    "IRI": "IRN", "ISV": "VIR", "IVB": "VGB", "KOS": "XKX", "KSA": "SAU",
    "KUW": "KWT", "LAT": "LVA", "LBA": "LBY", "LES": "LSO", "LIB": "LBN",
    "MAD": "MDG", "MAS": "MYS", "MAW": "MWI", "MGL": "MNG", "MON": "MCO",
    "MRI": "MUS", "MTN": "MRT", "MYA": "MMR", "NCA": "NIC", "NED": "NLD",
    "NEP": "NPL", "NGR": "NGA", "NIG": "NER", "OMA": "OMN", "PAR": "PRY",
    "PHI": "PHL", "PLE": "PSE", "POR": "PRT", "PUR": "PRI", "RSA": "ZAF",
    "SAM": "WSM", "SEY": "SYC", "SKN": "KNA", "SLO": "SVN", "SOL": "SLB",
    "SRI": "LKA", "SUD": "SDN", "SUI": "CHE", "TAN": "TZA", "TGA": "TON",
    "TOG": "TGO", "UAE": "ARE", "URU": "URY", "VAN": "VUT", "VIE": "VNM",
    "VIN": "VCT", "ZAM": "ZMB", "ZIM": "ZWE",
    # Same letters but a DIFFERENT country in the World Bank list, so fix by hand:
    "BRN": "BHR",  # Bahrain (ISO "BRN" is Brunei)
    "CHI": "CHL",  # Chile (World Bank "CHI" is the Channel Islands)
    # West Germany 1988: use Germany's figures.
    "FRG": "DEU",
    # Teams with no matching World Bank country:
    "AHO": None,  # Netherlands Antilles (dissolved)
    "COK": None,  # Cook Islands
    "EUN": None,  # Unified Team 1992 (ex-Soviet states)
    "GDR": None,  # East Germany
    "IOA": None,  # Independent Olympic Athletes
    "ROT": None,  # Refugee Olympic Team
    "SCG": None,  # Serbia and Montenegro
    "TCH": None,  # Czechoslovakia
    "TPE": None,  # Chinese Taipei (not a World Bank economy)
    "URS": None,  # Soviet Union
    "YAR": None, "YMD": None,  # Yemen (old North/South)
    "YUG": None,  # Yugoslavia
}

SERIES = {
    "GDP (current US$)": "GDP",
    "GDP per capita (current US$)": "GDP_per_capita",
    "GDP growth (annual %)": "GDP_growth",
    "Population, total": "Pop",
    "Population growth (annual %)": "Pop_growth",
    "Land area (sq. km)": "Area",
}


def load_wdi(path=RAW / "wdi_data.csv"):
    """Return one row per (ISO, Year) with the six indicators as columns."""
    w = pd.read_csv(path)
    w = w.dropna(subset=["Series Name"])  # drops blank/footer rows at the bottom
    w = w[w["Series Name"].isin(SERIES)]
    year_cols = [c for c in w.columns if "[YR" in c]
    long = w.melt(
        id_vars=["Country Code", "Series Name"], value_vars=year_cols,
        var_name="Year", value_name="value",
    )
    long["Year"] = long["Year"].str[:4].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")  # ".." becomes NaN
    long["Series Name"] = long["Series Name"].map(SERIES)
    wide = long.pivot_table(
        index=["Country Code", "Year"], columns="Series Name", values="value"
    ).reset_index()
    return wide.rename(columns={"Country Code": "ISO"})


def build_merged():
    medals = build_medal_table()
    wdi = load_wdi()
    medals["ISO"] = medals["NOC"].map(lambda c: NOC_TO_ISO.get(c, c))
    # Keep rows with no World Bank match for now (ISO may be None) so we can report them.
    merged = medals.merge(wdi, on=["ISO", "Year"], how="left")
    return merged


if __name__ == "__main__":
    m = build_merged()
    indicators = list(SERIES.values())
    print(m.head(10).to_string(index=False))
    print("\nrows:", len(m))

    no_iso = m[m["ISO"].isna()]
    print("\nTeams with no World Bank country (dropped later):")
    print(no_iso.groupby("NOC")[["Medals"]].sum().sort_values("Medals", ascending=False).to_string())

    has_iso = m[m["ISO"].notna()]
    unmatched = has_iso[has_iso[indicators].isna().all(axis=1)]
    print("\nCode mapped but World Bank has no data row at all:")
    print(unmatched[["Year", "NOC", "ISO", "Medals"]].to_string(index=False))

    print("\nMissing values per column (these rows get dropped in Step 6):")
    print(m[indicators].isna().sum().to_string())
    print("\nCheck, USA 2016:")
    print(m[(m.NOC == "USA") & (m.Year == 2016)].T.to_string(header=False))
