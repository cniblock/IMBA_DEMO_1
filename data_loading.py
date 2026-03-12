"""Data loading, caching, and district lookup for the STATS19 Intelligence Platform."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from config import DATASET_CANDIDATES, LAD_LOOKUP, LEGACY_LAD_CROSSWALK


def find_dataset_dir() -> Path:
    for candidate in DATASET_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find `dataset` or `datasets` directory in workspace.")


def data_cache_fingerprint() -> tuple:
    """Build a cache fingerprint from all relevant source files for correct invalidation."""
    dataset_dir = find_dataset_dir()
    files = [
        dataset_dir / "dft-road-casualty-statistics-collision-last-5-years.csv",
        dataset_dir / "dft-road-casualty-statistics-vehicle-last-5-years.csv",
        dataset_dir / "dft-road-casualty-statistics-casualty-last-5-years.csv",
        dataset_dir / "dft-road-casualty-statistics-collision-provisional-2025.csv",
        dataset_dir / "dft-road-casualty-statistics-vehicle-provisional-2025.csv",
        dataset_dir / "dft-road-casualty-statistics-casualty-provisional-2025.csv",
        dataset_dir / "Local_Authority_Districts_(April_2025)_Names_and_Codes_in_the_UK_v2.csv",
        dataset_dir / "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2024.xlsx",
    ]
    result = []
    for p in files:
        if p.exists():
            result.append((str(p), p.stat().st_mtime_ns, p.stat().st_size))
        else:
            result.append((str(p), None, None))
    return tuple(result)


@st.cache_data(show_spinner=False)
def get_lad_lookup() -> Dict[str, str]:
    lookup = dict(LAD_LOOKUP)
    try:
        dataset_dir = find_dataset_dir()
        lookup_path = dataset_dir / "Local_Authority_Districts_(April_2025)_Names_and_Codes_in_the_UK_v2.csv"
        if lookup_path.exists():
            lad_df = pd.read_csv(lookup_path, dtype="string", low_memory=False)
            if {"LAD25CD", "LAD25NM"}.issubset(set(lad_df.columns)):
                fresh = (
                    lad_df[["LAD25CD", "LAD25NM"]]
                    .dropna()
                    .drop_duplicates(subset=["LAD25CD"], keep="last")
                )
                fresh["LAD25CD"] = fresh["LAD25CD"].astype("string").str.strip().str.upper()
                lookup.update(dict(zip(fresh["LAD25CD"], fresh["LAD25NM"])))
    except Exception:
        pass
    return lookup


def add_district_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "local_authority_ons_district" not in out.columns:
        return out

    lad_lookup = get_lad_lookup()
    code = out["local_authority_ons_district"].astype("string").str.strip().str.upper()
    out["district_name"] = code.map(lad_lookup)
    out["district_current_authority"] = out["district_name"]
    for legacy_code, details in LEGACY_LAD_CROSSWALK.items():
        is_legacy = code == legacy_code
        if is_legacy.any():
            out.loc[is_legacy, "district_name"] = details["district_name"]
            out.loc[is_legacy, "district_current_authority"] = details["current_authority"]
    out["district_display"] = code.fillna("Unknown")
    mapped_mask = out["district_name"].notna()
    out.loc[mapped_mask, "district_display"] = (
        out.loc[mapped_mask, "district_name"] + " (" + code.loc[mapped_mask] + ")"
    )
    return out


def district_authority_lookup_frame(collision_view: pd.DataFrame) -> pd.DataFrame:
    """Build a safe district->current-authority lookup."""
    base = collision_view[["district_display"]].drop_duplicates().copy()
    if "district_current_authority" in collision_view.columns:
        auth = (
            collision_view[["district_display", "district_current_authority"]]
            .drop_duplicates(subset=["district_display"])
            .rename(columns={"district_current_authority": "Current Authority"})
        )
    else:
        auth = base.assign(**{"Current Authority": "Unknown"})
    return auth


def resolve_district_display(code_series: pd.Series) -> pd.Series:
    code = code_series.astype("string").str.strip().str.upper()
    lad_lookup = get_lad_lookup()
    name = code.map(lad_lookup)
    for legacy_code, details in LEGACY_LAD_CROSSWALK.items():
        name = name.mask(code == legacy_code, details["district_name"])
    display = code.fillna("Unknown")
    mapped_mask = name.notna()
    display.loc[mapped_mask] = name.loc[mapped_mask] + " (" + code.loc[mapped_mask] + ")"
    return display


def _load_and_concat(
    base_path: Path,
    provisional_path: Path,
    dtype_common: dict,
) -> pd.DataFrame:
    """Load base CSV and append provisional 2025 if present."""
    df = pd.read_csv(base_path, dtype=dtype_common, low_memory=False)
    if provisional_path.exists():
        prov = pd.read_csv(provisional_path, dtype=dtype_common, low_memory=False)
        df = pd.concat([df, prov], ignore_index=True, sort=False)
    return df


@st.cache_data(show_spinner=True)
def load_raw_tables(cache_fingerprint: tuple) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ = cache_fingerprint
    dataset_dir = find_dataset_dir()

    dtype_common = {
        "collision_index": "string",
        "lsoa_of_accident_location": "string",
        "lsoa_of_driver": "string",
        "lsoa_of_casualty": "string",
        "local_authority_ons_district": "string",
    }

    collisions = _load_and_concat(
        dataset_dir / "dft-road-casualty-statistics-collision-last-5-years.csv",
        dataset_dir / "dft-road-casualty-statistics-collision-provisional-2025.csv",
        dtype_common,
    )
    vehicles = _load_and_concat(
        dataset_dir / "dft-road-casualty-statistics-vehicle-last-5-years.csv",
        dataset_dir / "dft-road-casualty-statistics-vehicle-provisional-2025.csv",
        dtype_common,
    )
    casualties = _load_and_concat(
        dataset_dir / "dft-road-casualty-statistics-casualty-last-5-years.csv",
        dataset_dir / "dft-road-casualty-statistics-casualty-provisional-2025.csv",
        dtype_common,
    )

    for frame in [collisions, vehicles, casualties]:
        frame["collision_index"] = frame["collision_index"].astype("string")
    collisions["date"] = pd.to_datetime(collisions["date"], dayfirst=True, errors="coerce")
    collisions["time_dt"] = pd.to_datetime(collisions["time"], format="%H:%M", errors="coerce")
    collisions["hour"] = collisions["time_dt"].dt.hour
    collisions["is_dark"] = collisions["light_conditions"].isin([4, 5, 6, 7]).astype(int)
    collisions["is_weekend"] = collisions["day_of_week"].isin([1, 7]).astype(int)
    collisions = add_district_labels(collisions)

    return collisions, vehicles, casualties
