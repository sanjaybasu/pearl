"""
Area-level deprivation linkage.

Builds a ZIP-code-level Social Vulnerability Index (CDC/ATSDR, 2022) for the
cohort states and joins it to patients by residential ZIP from the health-plan
eligibility file.

Chain: patient -> ZIP5 -> ZCTA (identity where the ZCTA exists) -> census tracts
(Census 2020 ZCTA-to-tract relationship file) -> population-weighted mean of the
tract-level SVI overall percentile ranking.

Sources (public, no credential required):
  CDC/ATSDR SVI 2022 state files  https://svi.cdc.gov/Documents/Data/2022/csv/states/
  Census 2020 ZCTA-tract relationship file  https://www2.census.gov/geo/docs/maps-data/data/rel2020/

Writes data/processed/svi_by_zip.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXT = Path("/Users/sanjaybasu/waymark-local/data/external/svi")
OUT = Path("/Users/sanjaybasu/waymark-local/data/processed")
OUT.mkdir(parents=True, exist_ok=True)
RAW = Path("/Users/sanjaybasu/waymark-local/data/real_inputs")

STATES = ["Washington", "Virginia", "Ohio"]
# CDC codes missing values as -999
NA = -999


def load_svi() -> pd.DataFrame:
    frames = []
    for s in STATES:
        df = pd.read_csv(EXT / f"SVI2022_{s}.csv", low_memory=False)
        keep = ["ST_ABBR", "FIPS", "E_TOTPOP",
                "RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]
        df = df[keep].copy()
        frames.append(df)
    svi = pd.concat(frames, ignore_index=True)
    for c in ["RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]:
        svi[c] = pd.to_numeric(svi[c], errors="coerce").replace(NA, np.nan)
    svi["E_TOTPOP"] = pd.to_numeric(svi["E_TOTPOP"], errors="coerce").fillna(0)
    svi["tract"] = svi["FIPS"].astype(str).str.zfill(11)
    print(f"  SVI tracts loaded: {len(svi):,} "
          f"({svi['RPL_THEMES'].notna().sum():,} with an overall ranking)")
    return svi


def load_crosswalk() -> pd.DataFrame:
    rel = pd.read_csv(EXT / "zcta_tract_rel2020.txt", sep="|", dtype=str,
                      usecols=["GEOID_ZCTA5_20", "GEOID_TRACT_20"])
    rel = rel.dropna()
    rel.columns = ["zcta", "tract"]
    rel["zcta"] = rel["zcta"].str.zfill(5)
    rel["tract"] = rel["tract"].str.zfill(11)
    print(f"  ZCTA-tract pairs: {len(rel):,}")
    return rel


def zip5(series: pd.Series) -> pd.Series:
    """
    Normalise a mixed-format ZIP field to five digits.

    The eligibility extract stores ZIP+4 as nine digits for most Washington
    members and plain five digits for most Virginia members. Left-padding
    everything to nine before slicing would silently convert a five-digit
    Virginia ZIP such as 23294 into 00002, so length is resolved first.
    """
    s = series.astype(str).str.replace(r"\D", "", regex=True)
    out = np.where(s.str.len() >= 9, s.str[:5], s.str.zfill(5))
    out = pd.Series(out, index=series.index)
    return out.where(s.str.len() > 0, other=pd.NA)


def main():
    print("Building ZIP-level Social Vulnerability Index...")
    svi = load_svi()
    rel = load_crosswalk()

    m = rel.merge(svi[["tract", "E_TOTPOP", "RPL_THEMES", "RPL_THEME1",
                       "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]],
                  on="tract", how="inner")
    m = m[m["RPL_THEMES"].notna()]
    print(f"  matched tract-ZCTA rows in cohort states: {len(m):,}")

    # population-weighted mean of tract percentile rankings within each ZCTA
    def wmean(g, col):
        w = g["E_TOTPOP"].values
        v = g[col].values
        return np.average(v, weights=w) if w.sum() > 0 else np.nan

    rows = []
    for zcta, g in m.groupby("zcta"):
        rows.append({
            "zip5": zcta,
            "svi_overall": wmean(g, "RPL_THEMES"),
            "svi_socioeconomic": wmean(g, "RPL_THEME1"),
            "svi_household": wmean(g, "RPL_THEME2"),
            "svi_minority_language": wmean(g, "RPL_THEME3"),
            "svi_housing_transport": wmean(g, "RPL_THEME4"),
            "svi_population": float(g["E_TOTPOP"].sum()),
            "svi_n_tracts": int(len(g)),
        })
    zip_svi = pd.DataFrame(rows)
    zip_svi["svi_quintile"] = pd.qcut(zip_svi["svi_overall"], 5,
                                      labels=[1, 2, 3, 4, 5]).astype(int)
    zip_svi.to_parquet(OUT / "svi_by_zip.parquet", index=False)
    print(f"  ZIPs with an SVI value: {len(zip_svi):,}")
    print(zip_svi["svi_overall"].describe().round(3).to_string())

    # coverage against the cohort's residential ZIPs
    elig = pd.read_parquet(RAW / "eligibility.parquet",
                           columns=["member_id", "zip_code", "state"])
    elig = elig.drop_duplicates("member_id")
    elig["zip5"] = zip5(elig["zip_code"])
    cov = elig.merge(zip_svi[["zip5", "svi_overall"]], on="zip5", how="left")
    print(f"\n  eligibility members: {len(cov):,}")
    print(f"  with an SVI-linked ZIP: {cov['svi_overall'].notna().sum():,} "
          f"({100*cov['svi_overall'].notna().mean():.1f}%)")
    print(cov.groupby("state")["svi_overall"]
             .agg(["size", "count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
