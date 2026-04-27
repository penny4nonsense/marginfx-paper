"""
clean_adult.py
--------------
Data cleaning script for the UCI Adult Income dataset.

Loads raw adult.data and adult.test, cleans and processes them,
and saves the processed dataset to data/processed/adult/.

Feature groups for specification search:

    Group A — Demographics (core):
        age, female, education_num

    Group B — Work characteristics:
        hours_per_week, government, self_employed,
        white_collar, blue_collar, service

    Group C — Financial:
        capital_gain, capital_loss, married

Outcome:
    income — 1 if income > $50k, 0 otherwise

Categorical encoding:
    workclass   -> government, self_employed (private = reference, dropped)
    occupation  -> white_collar, blue_collar, service (sales = reference, dropped)
    marital     -> married (not married = reference, dropped)

Four model specifications:
    spec_A:     Group A only
    spec_AB:    Group A + Group B
    spec_AC:    Group A + Group C
    spec_ABC:   Group A + Group B + Group C (full model)

Usage:
    python clean_adult.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(SCRIPT_DIR, '..', '..')
RAW_DIR = os.path.join(PAPER_DIR, 'data', 'raw', 'adult')
PROCESSED_DIR = os.path.join(PAPER_DIR, 'data', 'processed', 'adult')

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

COL_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income'
]

# ---------------------------------------------------------------------------
# Categorical groupings
# ---------------------------------------------------------------------------

WORKCLASS_GOVERNMENT = ['Federal-gov', 'State-gov', 'Local-gov']
WORKCLASS_SELF_EMPLOYED = ['Self-emp-inc', 'Self-emp-not-inc']
# Reference: Private

OCCUPATION_WHITE_COLLAR = [
    'Exec-managerial', 'Prof-specialty', 'Tech-support', 'Adm-clerical'
]
OCCUPATION_BLUE_COLLAR = [
    'Craft-repair', 'Machine-op-inspct', 'Transport-moving',
    'Handlers-cleaners', 'Farming-fishing'
]
OCCUPATION_SERVICE = [
    'Other-service', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'
]
# Reference: Sales

MARITAL_MARRIED = ['Married-civ-spouse', 'Married-AF-spouse']
# Reference: all others (never married, divorced, separated, widowed)

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

GROUP_A = ['age', 'female', 'education_num']

GROUP_B = [
    'hours_per_week', 'government', 'self_employed',
    'white_collar', 'blue_collar', 'service'
]

GROUP_C = ['capital_gain', 'capital_loss', 'married']

OUTCOME = 'income'

# Categorical features for marginfx (binary indicators)
CATEGORICAL_FEATURES = [
    'female', 'government', 'self_employed',
    'white_collar', 'blue_collar', 'service', 'married'
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """
    Load and concatenate adult.data and adult.test.

    Returns
    -------
    pd.DataFrame
        Combined raw dataset.
    """
    train_path = os.path.join(RAW_DIR, 'adult.data')
    test_path = os.path.join(RAW_DIR, 'adult.test')

    df_train = pd.read_csv(
        train_path,
        header=None,
        names=COL_NAMES,
        sep=',',
        skipinitialspace=True,
        na_values='?',
    )

    df_test = pd.read_csv(
        test_path,
        header=None,
        names=COL_NAMES,
        sep=',',
        skipinitialspace=True,
        na_values='?',
        skiprows=1,  # test file has a header comment line
    )

    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Loaded {len(df):,} rows ({len(df_train):,} train + {len(df_test):,} test)")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from raw Adult dataset.

    Steps:
        1. Strip whitespace from string columns
        2. Clean income column (test set has trailing period)
        3. Create binary outcome
        4. Create binary sex indicator
        5. Create grouped categorical indicators
        6. Drop rows with missing values
        7. Select and return final features

    Parameters
    ----------
    df : pd.DataFrame
        Raw combined dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with all features and outcome.
    """
    df = df.copy()

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Clean income — test set has trailing period ('>50K.')
    df['income'] = df['income'].str.rstrip('.')

    # Binary outcome
    df[OUTCOME] = (df['income'] == '>50K').astype(int)

    # --- Group A: Demographics ---
    df['female'] = (df['sex'] == 'Female').astype(int)
    # age and education_num already numeric

    # --- Group B: Work characteristics ---
    # workclass groupings
    df['government'] = df['workclass'].isin(WORKCLASS_GOVERNMENT).astype(int)
    df['self_employed'] = df['workclass'].isin(WORKCLASS_SELF_EMPLOYED).astype(int)

    # occupation groupings
    df['white_collar'] = df['occupation'].isin(OCCUPATION_WHITE_COLLAR).astype(int)
    df['blue_collar'] = df['occupation'].isin(OCCUPATION_BLUE_COLLAR).astype(int)
    df['service'] = df['occupation'].isin(OCCUPATION_SERVICE).astype(int)

    # --- Group C: Financial ---
    df['married'] = df['marital_status'].isin(MARITAL_MARRIED).astype(int)
    # capital_gain and capital_loss already numeric

    # Drop missing values
    all_features = GROUP_A + GROUP_B + GROUP_C + [OUTCOME]
    df = df[all_features].dropna()

    print(f"After cleaning: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Outcome prevalence: {df[OUTCOME].mean():.1%} earn > $50k")

    return df


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame) -> None:
    """
    Save processed dataset and feature group metadata.

    Files saved:
        adult_processed.csv     — full processed dataset
        feature_groups.py       — feature group definitions for analysis scripts
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save processed data
    out_path = os.path.join(PROCESSED_DIR, 'adult_processed.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved processed data: {out_path}")

    # Save feature group definitions as a Python file
    # Analysis scripts import this directly
    groups_path = os.path.join(PROCESSED_DIR, 'feature_groups.py')
    with open(groups_path, 'w') as f:
        f.write('"""\nFeature group definitions for UCI Adult Income dataset.\nAuto-generated by clean_adult.py\n"""\n\n')
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
    print("UCI Adult Income — Processed Dataset Summary")
    print("=" * 55)
    print(f"\nObservations: {len(df):,}")
    print(f"Outcome: {df[OUTCOME].mean():.1%} earn > $50k")

    print("\nGroup A — Demographics:")
    print(df[GROUP_A].describe().round(2).to_string())

    print("\nGroup B — Work characteristics:")
    print(df[GROUP_B].describe().round(2).to_string())

    print("\nGroup C — Financial:")
    print(df[GROUP_C].describe().round(2).to_string())

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
    print("Cleaning UCI Adult Income dataset...")
    print()

    df_raw = load_raw()
    df_clean = clean(df_raw)
    print_summary(df_clean)
    save(df_clean)

    print("\nDone.")