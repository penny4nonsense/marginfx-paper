"""
clean_ames_housing.py
---------------------
Data cleaning script for the Ames Housing dataset.

Loads raw ames_housing.csv, cleans and processes it,
and saves the processed dataset to data/processed/ames_housing/.

Feature groups for specification search:

    Group A — Core physical characteristics:
        GrLivArea       — above ground living area (sq ft)
        BedroomAbvGr    — number of bedrooms above ground
        FullBath        — number of full bathrooms
        HalfBath        — number of half bathrooms

    Group B — Quality and condition:
        OverallQual     — overall material and finish quality (1-10)
        OverallCond     — overall condition rating (1-10)
        YearBuilt       — year built
        YearRemodAdd    — year remodeled (same as YearBuilt if no remodel)

    Group C — Lot and location:
        LotArea         — lot size in square feet
        GarageArea      — garage size in square feet
        nbhd_high       — 1 if top tier neighborhood
        nbhd_mid        — 1 if middle tier neighborhood
                          (bottom tier = reference, dropped)

Outcome:
    SalePrice — sale price in dollars (regression)

Categorical features for marginfx:
    nbhd_high, nbhd_mid

Neighborhood grouping:
    Neighborhoods grouped into three tiers by median sale price.
    Top tier (nbhd_high): NridgHt, NoRidge, StoneBr, Timber, Veenker
    Mid tier (nbhd_mid):  Somerst, CollgCr, ClearCr, Crawfor, Blmngtn,
                          Gilbert, NWAmes, SawyerW, Mitchel
    Bottom tier (ref):    all others

Four model specifications:
    spec_A:     Group A only
    spec_AB:    Group A + Group B
    spec_AC:    Group A + Group C
    spec_ABC:   Group A + Group B + Group C (full model)

Usage:
    python clean_ames_housing.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(SCRIPT_DIR, '..', '..')
RAW_DIR = os.path.join(PAPER_DIR, 'data', 'raw', 'ames_housing')
PROCESSED_DIR = os.path.join(PAPER_DIR, 'data', 'processed', 'ames_housing')

# ---------------------------------------------------------------------------
# Neighborhood tiers
# (grouped by median sale price from the full dataset)
# ---------------------------------------------------------------------------

NBHD_HIGH = [
    'NridgHt', 'NoRidge', 'StoneBr', 'Timber', 'Veenker'
]

NBHD_MID = [
    'Somerst', 'CollgCr', 'ClearCr', 'Crawfor', 'Blmngtn',
    'Gilbert', 'NWAmes', 'SawyerW', 'Mitchel'
]

# Bottom tier = reference category (dropped)
# All neighborhoods not in HIGH or MID

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

GROUP_A = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']

GROUP_B = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']

GROUP_C = ['LotArea', 'GarageArea', 'nbhd_high', 'nbhd_mid']

OUTCOME = 'SalePrice'

CATEGORICAL_FEATURES = ['nbhd_high', 'nbhd_mid']


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """
    Load raw Ames Housing dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    raw_path = os.path.join(RAW_DIR, 'ames_housing.csv')
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from raw Ames Housing dataset.

    Steps:
        1. Select relevant raw columns
        2. Create neighborhood tier indicators
        3. Handle missing values
        4. Select and return final features

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with all features and outcome.
    """
    df = df.copy()

    # Check which columns are available
    available = df.columns.tolist()

    # Handle different possible column name formats
    # Some versions of the dataset use 'Id' others don't have it
    # SalePrice may be the target

    # Required raw columns
    required = [
        'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
        'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'LotArea', 'GarageArea', 'Neighborhood', 'SalePrice'
    ]

    missing_cols = [c for c in required if c not in available]
    if missing_cols:
        print(f"Warning: missing columns: {missing_cols}")
        print(f"Available columns: {available[:20]}...")

    # --- Group C: Neighborhood tier indicators ---
    if 'Neighborhood' in df.columns:
        df['nbhd_high'] = df['Neighborhood'].isin(NBHD_HIGH).astype(int)
        df['nbhd_mid'] = df['Neighborhood'].isin(NBHD_MID).astype(int)
    else:
        df['nbhd_high'] = 0
        df['nbhd_mid'] = 0

    # Handle GarageArea missing values — fill with 0 (no garage)
    if 'GarageArea' in df.columns:
        df['GarageArea'] = df['GarageArea'].fillna(0)

    # Select final features
    all_features = GROUP_A + GROUP_B + GROUP_C + [OUTCOME]
    available_features = [f for f in all_features if f in df.columns]
    df = df[available_features].dropna()

    print(f"After cleaning: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Sale price: ${df[OUTCOME].mean():,.0f} mean, "
          f"${df[OUTCOME].median():,.0f} median")

    return df


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame) -> None:
    """
    Save processed dataset and feature group metadata.

    Files saved:
        ames_housing_processed.csv  — full processed dataset
        feature_groups.py           — feature group definitions
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save processed data
    out_path = os.path.join(PROCESSED_DIR, 'ames_housing_processed.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved processed data: {out_path}")

    # Save feature group definitions
    groups_path = os.path.join(PROCESSED_DIR, 'feature_groups.py')
    with open(groups_path, 'w') as f:
        f.write('"""\nFeature group definitions for Ames Housing dataset.\nAuto-generated by clean_ames_housing.py\n"""\n\n')
        f.write(f'GROUP_A = {GROUP_A}\n')
        f.write(f'GROUP_B = {GROUP_B}\n')
        f.write(f'GROUP_C = {GROUP_C}\n')
        f.write(f'OUTCOME = "{OUTCOME}"\n')
        f.write(f'CATEGORICAL_FEATURES = {CATEGORICAL_FEATURES}\n')
        f.write('\n')
        f.write('SPECIFICATIONS = {\n')
        f.write('    "A":   GROUP_A,\n')
        f.write('    "AB":  GROUP_A + GROUP_B,\n')
        f.write('    "AC":  GROUP_A + GROUP_C,\n')
        f.write('    "ABC": GROUP_A + GROUP_B + GROUP_C,\n')
        f.write('}\n')

    print(f"Saved feature groups: {groups_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the processed dataset."""
    print("\n" + "=" * 55)
    print("Ames Housing — Processed Dataset Summary")
    print("=" * 55)
    print(f"\nObservations: {len(df):,}")
    print(f"Sale price: ${df[OUTCOME].mean():,.0f} mean, "
          f"${df[OUTCOME].median():,.0f} median, "
          f"${df[OUTCOME].std():,.0f} std")

    print("\nGroup A — Core physical characteristics:")
    print(df[GROUP_A].describe().round(2).to_string())

    print("\nGroup B — Quality and condition:")
    print(df[GROUP_B].describe().round(2).to_string())

    print("\nGroup C — Lot and location:")
    print(df[GROUP_C].describe().round(2).to_string())

    print(f"\nNeighborhood tiers:")
    print(f"  High tier:   {df['nbhd_high'].sum():,} homes ({df['nbhd_high'].mean():.1%})")
    print(f"  Mid tier:    {df['nbhd_mid'].sum():,} homes ({df['nbhd_mid'].mean():.1%})")
    print(f"  Bottom tier: {(1 - df['nbhd_high'] - df['nbhd_mid']).sum():,} homes "
          f"({(1 - df['nbhd_high'] - df['nbhd_mid']).mean():.1%})")

    print("\nSpecifications:")
    for spec, features in [
        ('A',   GROUP_A),
        ('AB',  GROUP_A + GROUP_B),
        ('AC',  GROUP_A + GROUP_C),
        ('ABC', GROUP_A + GROUP_B + GROUP_C),
    ]:
        print(f"  Spec {spec}: {len(features)} features — {features}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Cleaning Ames Housing dataset...")
    print()

    df_raw = load_raw()
    df_clean = clean(df_raw)
    print_summary(df_clean)
    save(df_clean)

    print("\nDone.")
