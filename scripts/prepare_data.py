#!/usr/bin/env python3
"""
Convert CSV to Parquet and prebuild views for faster app loading.
Run once after cloning or when source data changes:
    python scripts/prepare_data.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _to_parquet_safe(df, path):
    """Save DataFrame to Parquet, converting category columns to string for PyArrow compatibility."""
    out = df.copy()
    for col in out.select_dtypes(include=["category"]).columns:
        out[col] = out[col].astype(str)
    out.to_parquet(path, index=False)


def main():
    from data_loading import data_cache_fingerprint, get_parquet_cache_dir, load_raw_tables
    from transforms import build_collision_view, build_vehicle_view, build_casualty_views

    fp = data_cache_fingerprint()
    cache_dir = get_parquet_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw tables (CSV to Parquet on first run)...")
    load_raw_tables(fp)
    print("Building collision view...")
    cv = build_collision_view(fp)
    _to_parquet_safe(cv, cache_dir / "collision_view.parquet")
    print("Building vehicle view...")
    vv = build_vehicle_view(fp)
    _to_parquet_safe(vv, cache_dir / "vehicle_view.parquet")
    print("Building casualty views...")
    casualty_view, casualty_person_view = build_casualty_views(fp)
    _to_parquet_safe(casualty_view, cache_dir / "casualty_view.parquet")
    _to_parquet_safe(casualty_person_view, cache_dir / "casualty_person_view.parquet")
    print(f"Done. Prebuilt views saved to {cache_dir}")

if __name__ == "__main__":
    main()
