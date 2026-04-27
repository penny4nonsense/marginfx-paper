"""
clean_credit_default.py
-----------------------
Data cleaning script for the UCI Credit Card Default dataset.

Loads raw credit_default.xls, cleans and processes it,
and saves the processed dataset to data/processed/credit_default/.

Feature groups for specification search:

    Group A — Demographics:
        age, female, education, married

    Group B — Payment history (summarized):
        avg_pay_status  — mean repayment status across 6 months
        months_delayed  — number of months with late payment (PAY > 0)

    Group C — Bill and payment amounts (summarized):
        avg_bill_amt    — mean bill amount across 6 months
        avg_pay_amt     — mean payment amount across 6 months
        pay_ratio       — avg_pay_amt / (avg_bill_amt + 1)

Outcome:
    default — 1 if defaulted next month, 0 otherwise

Categorical features for marginfx:
    female, married

Four model specifications:
    spec_A:     Group A only
    spec_AB:    Group A + Group B
    spec_AC:    Group A + Group C
    spec_ABC:   Group A + Group B + Group C (full model)

Usage:
    python clean_credit_default.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(SCRIPT_DIR, '..', '..')
RAW_DIR = os.path.join(PAPER_DIR, 'data', 'raw', 'credit_default')
PROCESSED_DIR = os.path.join(PAPER_DIR, 'data', 'processed', 'credit_default')

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

GROUP_A = ['age', 'female', 'education', 'married']

GROUP_B = ['avg_pay_status', 'months_delayed']

GROUP_C = ['avg_bill_amt', 'avg_pay_amt', 'pay_ratio']

OUTCOME = 'default'

CATEGORICAL_FEATURES = ['female', 'married']

# Payment status columns (PAY_0 is most recent)
PAY_COLS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

# Bill amount columns
BILL_COLS = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
             'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

# Payment amount columns
PAY_AMT_COLS = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """
    Load raw Credit Card Default dataset from XLS file.

    The dataset has a header row at row 0 and column names at row 1.
    We skip row 0 and use row 1 as the header.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    raw_path = os.path.join(RAW_DIR, 'credit_default.xls')

    df = pd.read_excel(raw_path, header=1)

    # Drop the ID column if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from raw Credit Card Default dataset.

    Steps:
        1. Rename outcome column
        2. Create binary demographic indicators
        3. Clean education and marriage codes
        4. Summarize payment history into Group B features
        5. Summarize bill/payment amounts into Group C features
        6. Drop missing values
        7. Select and return final features

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

    # Rename outcome column
    # Column is named 'default.payment.next.month' or similar
    outcome_col = [c for c in df.columns if 'default' in c.lower()][0]
    df[OUTCOME] = df[outcome_col]

    # --- Group A: Demographics ---

    # Age — already numeric
    df['age'] = df['AGE']

    # Female indicator — SEX: 1=male, 2=female
    df['female'] = (df['SEX'] == 2).astype(int)

    # Education — EDUCATION: 1=graduate, 2=university, 3=high school, 4=others
    # Treat as ordinal, recode 0, 5, 6 (undocumented) as 4 (other)
    df['education'] = df['EDUCATION'].copy()
    df.loc[df['education'].isin([0, 5, 6]), 'education'] = 4

    # Married — MARRIAGE: 1=married, 2=single, 3=others
    # Binary: 1 if married, 0 otherwise
    df['married'] = (df['MARRIAGE'] == 1).astype(int)

    # --- Group B: Payment history (summarized) ---

    # Average payment status across 6 months
    # PAY values: -2=no consumption, -1=paid in full, 0=minimum paid,
    #              1=one month late, 2=two months late, etc.
    df['avg_pay_status'] = df[PAY_COLS].mean(axis=1)

    # Number of months with late payment (PAY > 0)
    df['months_delayed'] = (df[PAY_COLS] > 0).sum(axis=1)

    # --- Group C: Bill and payment amounts (summarized) ---

    # Average bill amount across 6 months (in NT dollars)
    df['avg_bill_amt'] = df[BILL_COLS].mean(axis=1)

    # Average payment amount across 6 months
    df['avg_pay_amt'] = df[PAY_AMT_COLS].mean(axis=1)

    # Payment ratio — how much of the bill is being paid on average
    # Add 1 to denominator to avoid division by zero
    df['pay_ratio'] = df['avg_pay_amt'] / (df['avg_bill_amt'].abs() + 1)

    # Clip pay_ratio to reasonable range [0, 1]
    df['pay_ratio'] = df['pay_ratio'].clip(0, 1)

    # Select final features
    all_features = GROUP_A + GROUP_B + GROUP_C + [OUTCOME]
    df = df[all_features].dropna()

    print(f"After cleaning: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Outcome prevalence: {df[OUTCOME].mean():.1%} defaulted")

    return df


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame) -> None:
    """
    Save processed dataset and feature group metadata.

    Files saved:
        credit_default_processed.csv    — full processed dataset
        feature_groups.py               — feature group definitions
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save processed data
    out_path = os.path.join(PROCESSED_DIR, 'credit_default_processed.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved processed data: {out_path}")

    # Save feature group definitions
    groups_path = os.path.join(PROCESSED_DIR, 'feature_groups.py')
    with open(groups_path, 'w') as f:
        f.write('"""\nFeature group definitions for UCI Credit Card Default dataset.\nAuto-generated by clean_credit_default.py\n"""\n\n')
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
    print("UCI Credit Card Default — Processed Dataset Summary")
    print("=" * 55)
    print(f"\nObservations: {len(df):,}")
    print(f"Outcome: {df[OUTCOME].mean():.1%} defaulted next month")

    print("\nGroup A — Demographics:")
    print(df[GROUP_A].describe().round(2).to_string())

    print("\nGroup B — Payment history:")
    print(df[GROUP_B].describe().round(2).to_string())

    print("\nGroup C — Bill and payment amounts:")
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
    print("Cleaning UCI Credit Card Default dataset...")
    print()

    df_raw = load_raw()
    df_clean = clean(df_raw)
    print_summary(df_clean)
    save(df_clean)

    print("\nDone.")
