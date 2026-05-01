"""
generate_tables.py
------------------
Generate LaTeX tables and summary parquet files for the marginfx paper.

For each empirical dataset (adult, credit_default, ames_housing), produces:

    Table 1 — Specification search (2 models x 4 specs):
        Rows: features grouped by feature group with panel labels
        Columns: 2 models x 4 specifications
        Each cell: estimate with significance stars, (SE) below
        Horizontal rules between feature groups
        Dataset-specific model pairs:
            adult:          logistic + xgboost
            credit_default: logistic + tensorflow
            ames_housing:   linear   + rf

    Table 2 — Model comparison (4 models, full spec ABC):
        Rows: features grouped with horizontal rule separators
        Columns: 4 models
        Each cell: estimate with significance stars, (SE) below
        Full specification only

    Table 3 — Method comparison (4 models x 3 methods):
        Rows: features grouped with horizontal rule separators
        Columns: 4 models x 3 methods (AME, SHAP, PDP)
        Each cell: point estimate only, no SEs, no stars

Formatting conventions:
    - booktabs rules: \\toprule (thick) / \\midrule / \\bottomrule (thick)
    - Panel labels in italic spanning full width
    - \\midrule between feature groups
    - Stars: *** p<0.01, ** p<0.05, * p<0.10
    - SEs in parentheses below estimates

Usage:
    python generate_tables.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = SCRIPT_DIR
TABLES_DIR = os.path.join(PAPER_DIR, 'tables')
EMPIRICAL_DIR = os.path.join(PAPER_DIR, 'empirical')

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

RESULTS = {
    'adult': {
        'ames_path':  os.path.join(EMPIRICAL_DIR, 'adult', 'results', 'adult_ames.parquet'),
        'shap_path':  os.path.join(EMPIRICAL_DIR, 'adult', 'results', 'adult_shap.parquet'),
        'pdp_path':   os.path.join(EMPIRICAL_DIR, 'adult', 'results', 'adult_pdp.parquet'),
        'outcome_type': 'classification',
        'spec_models': ['logistic', 'xgboost'],
        'all_models':  ['logistic', 'rf', 'xgboost', 'tensorflow'],
        'outcome_label': r'P(Income $>$ \$50k)',
        'dataset_label': 'UCI Adult Income',
        'feature_groups': {
            'Panel A: Demographics': ['age', 'female', 'education_num'],
            'Panel B: Work Characteristics': [
                'hours_per_week', 'government', 'self_employed',
                'white_collar', 'blue_collar', 'service'
            ],
            'Panel C: Financial': ['capital_gain', 'capital_loss', 'married'],
        },
    },
    'credit_default': {
        'ames_path':  os.path.join(EMPIRICAL_DIR, 'credit_default', 'results', 'credit_default_ames.parquet'),
        'shap_path':  os.path.join(EMPIRICAL_DIR, 'credit_default', 'results', 'credit_default_shap.parquet'),
        'pdp_path':   os.path.join(EMPIRICAL_DIR, 'credit_default', 'results', 'credit_default_pdp.parquet'),
        'outcome_type': 'classification',
        'spec_models': ['logistic', 'tensorflow'],
        'all_models':  ['logistic', 'rf', 'xgboost', 'tensorflow'],
        'outcome_label': 'P(Default)',
        'dataset_label': 'UCI Credit Card Default',
        'feature_groups': {
            'Panel A: Demographics': ['age', 'female', 'education', 'married'],
            'Panel B: Payment History': ['avg_pay_status', 'months_delayed'],
            'Panel C: Bill and Payment Amounts': ['avg_bill_amt', 'avg_pay_amt', 'pay_ratio'],
        },
    },
    'ames_housing': {
        'ames_path':  os.path.join(EMPIRICAL_DIR, 'ames_housing', 'results', 'ames_housing_ames.parquet'),
        'shap_path':  os.path.join(EMPIRICAL_DIR, 'ames_housing', 'results', 'ames_housing_shap.parquet'),
        'pdp_path':   os.path.join(EMPIRICAL_DIR, 'ames_housing', 'results', 'ames_housing_pdp.parquet'),
        'outcome_type': 'regression',
        'spec_models': ['linear', 'rf'],
        'all_models':  ['linear', 'rf', 'xgboost', 'tensorflow'],
        'outcome_label': 'Sale Price (USD)',
        'dataset_label': 'Ames Housing',
        'feature_groups': {
            'Panel A: Physical Characteristics': [
                'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath'
            ],
            'Panel B: Quality and Condition': [
                'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd'
            ],
            'Panel C: Lot and Location': [
                'LotArea', 'GarageArea', 'nbhd_high', 'nbhd_mid'
            ],
        },
    },
}

MODEL_LABELS = {
    'logistic':   'Logistic',
    'linear':     'Linear',
    'rf':         'Random Forest',
    'xgboost':    'XGBoost',
    'tensorflow': 'Neural Net',
}

SPEC_LABELS = {
    'A':   'Spec A',
    'AB':  'A+B',
    'AC':  'A+C',
    'ABC': 'A+B+C',
}

SPECS_ORDER = ['A', 'AB', 'AC', 'ABC']


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ''
    if p_value < 0.01:
        return '$^{***}$'
    elif p_value < 0.05:
        return '$^{**}$'
    elif p_value < 0.10:
        return '$^{*}$'
    return ''


def fmt_estimate(estimate: float, p_value: float, decimals: int = 4) -> str:
    if pd.isna(estimate):
        return '--'
    return f"{estimate:.{decimals}f}{stars(p_value)}"


def fmt_se(se: float, decimals: int = 4) -> str:
    if pd.isna(se):
        return ''
    return f"({se:.{decimals}f})"


def fmt_point(value: float, decimals: int = 4) -> str:
    if pd.isna(value):
        return '--'
    return f"{value:.{decimals}f}"


def get_decimals(outcome_type: str) -> int:
    return 2 if outcome_type == 'regression' else 4


def escape_feature(name: str) -> str:
    """Escape underscores and format feature names for LaTeX."""
    return name.replace('_', '\\_')


def panel_label_row(label: str, n_cols: int) -> str:
    """Generate an italic panel label spanning all columns."""
    return f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textit{{{label}}}}} \\\\"


# ---------------------------------------------------------------------------
# LaTeX table wrappers
# ---------------------------------------------------------------------------

def latex_begin(caption: str, label: str, col_spec: str) -> list:
    return [
        r'\begin{table*}[htbp]',
        r'\centering',
        r'\small',
        r'\begin{threeparttable}',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
    ]


def latex_end_with_stars() -> list:
    return [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item \textit{Notes:} $^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.10. '
        r'Standard errors from nonparametric bootstrap (200 replicates) in parentheses.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table*}',
    ]


def latex_end_no_stars() -> list:
    return [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item \textit{Notes:} AME = Average Marginal Effect (marginfx). '
        r'SHAP = mean signed SHAP value (GradientExplainer). '
        r'PDP = slope of partial dependence plot. '
        r'Full specification (A+B+C) only. No standard errors reported for SHAP and PDP.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table*}',
    ]


# ---------------------------------------------------------------------------
# Feature group iteration helper
# ---------------------------------------------------------------------------

def iter_feature_groups(feature_groups: dict, available_features: list):
    """
    Yield (panel_label, features_in_panel) for features that exist in data.
    Filters out features not present in available_features.
    """
    for panel_label, features in feature_groups.items():
        available = [f for f in features if f in available_features]
        if available:
            yield panel_label, available


# ---------------------------------------------------------------------------
# Table 1: Specification search
# ---------------------------------------------------------------------------

def make_table1(ames_df, dataset, config):
    decimals = get_decimals(config['outcome_type'])
    models = config['spec_models']
    n_data_cols = len(models) * len(SPECS_ORDER)
    n_total_cols = n_data_cols

    col_spec = 'l' + 'r' * n_data_cols
    caption = (
        f"{config['dataset_label']}: Specification Search. "
        f"Average marginal effects on {config['outcome_label']}."
    )
    label = f"tab:{dataset}_spec_search"

    lines = latex_begin(caption, label, col_spec)

    # Model header
    model_header = ''
    for model in models:
        model_header += f' & \\multicolumn{{{len(SPECS_ORDER)}}}{{c}}{{{MODEL_LABELS[model]}}}'
    model_header += r' \\'
    lines.append(model_header)

    # Cmidrule under model names
    cmidrule = ''
    col = 2
    for _ in models:
        cmidrule += f'\\cmidrule(lr){{{col}-{col + len(SPECS_ORDER) - 1}}} '
        col += len(SPECS_ORDER)
    lines.append(cmidrule)

    # Spec header
    spec_header = ''
    for _ in models:
        for spec in SPECS_ORDER:
            spec_header += f' & {SPEC_LABELS[spec]}'
    spec_header += r' \\'
    lines.append(spec_header)
    lines.append(r'\midrule')

    # Get available features
    available_features = ames_df['term'].unique().tolist()
    feature_groups = config['feature_groups']

    # Build summary data
    rows = []
    first_panel = True

    for panel_label, features in iter_feature_groups(feature_groups, available_features):
        # Panel separator and label
        if not first_panel:
            lines.append(r'\midrule')
        first_panel = False
        lines.append(panel_label_row(panel_label, n_data_cols))

        for feature in features:
            row = {'feature': feature}
            est_cells = []
            se_cells = []

            for model in models:
                for spec in SPECS_ORDER:
                    subset = ames_df[
                        (ames_df['term'] == feature) &
                        (ames_df['model'] == model) &
                        (ames_df['spec'] == spec)
                    ]
                    if len(subset) == 0:
                        est = se = p = np.nan
                    else:
                        est = subset['estimate'].values[0]
                        se = subset['std_error'].values[0]
                        p = subset['p_value'].values[0]

                    est_cells.append(fmt_estimate(est, p, decimals))
                    se_cells.append(fmt_se(se, decimals))

                    row[f'{model}_{spec}_est'] = est
                    row[f'{model}_{spec}_se'] = se
                    row[f'{model}_{spec}_p'] = p

            rows.append(row)

            # Estimate row
            est_row = f'\\quad {escape_feature(feature)}'
            for cell in est_cells:
                est_row += f' & {cell}'
            est_row += r' \\'
            lines.append(est_row)

            # SE row
            se_row = ''
            for cell in se_cells:
                se_row += f' & {cell}'
            se_row += r' \\'
            lines.append(se_row)

    lines.extend(latex_end_with_stars())

    summary_df = pd.DataFrame(rows)
    return '\n'.join(lines), summary_df


# ---------------------------------------------------------------------------
# Table 2: Model comparison
# ---------------------------------------------------------------------------

def make_table2(ames_df, dataset, config):
    decimals = get_decimals(config['outcome_type'])
    models = config['all_models']
    n_data_cols = len(models)

    col_spec = 'l' + 'r' * n_data_cols
    caption = (
        f"{config['dataset_label']}: Model Comparison. "
        f"Average marginal effects on {config['outcome_label']}, "
        f"full specification (A+B+C)."
    )
    label = f"tab:{dataset}_model_comparison"

    lines = latex_begin(caption, label, col_spec)

    # Header
    header = 'Feature'
    for model in models:
        header += f' & {MODEL_LABELS[model]}'
    header += r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    full_df = ames_df[ames_df['spec'] == 'ABC']
    available_features = full_df['term'].unique().tolist()
    feature_groups = config['feature_groups']

    rows = []
    first_panel = True

    for panel_label, features in iter_feature_groups(feature_groups, available_features):
        if not first_panel:
            lines.append(r'\midrule')
        first_panel = False

        for feature in features:
            row = {'feature': feature}
            est_cells = []
            se_cells = []

            for model in models:
                subset = full_df[
                    (full_df['term'] == feature) &
                    (full_df['model'] == model)
                ]
                if len(subset) == 0:
                    est = se = p = np.nan
                else:
                    est = subset['estimate'].values[0]
                    se = subset['std_error'].values[0]
                    p = subset['p_value'].values[0]

                est_cells.append(fmt_estimate(est, p, decimals))
                se_cells.append(fmt_se(se, decimals))
                row[f'{model}_est'] = est
                row[f'{model}_se'] = se
                row[f'{model}_p'] = p

            rows.append(row)

            est_row = escape_feature(feature)
            for cell in est_cells:
                est_row += f' & {cell}'
            est_row += r' \\'
            lines.append(est_row)

            se_row = ''
            for cell in se_cells:
                se_row += f' & {cell}'
            se_row += r' \\'
            lines.append(se_row)

    lines.extend(latex_end_with_stars())
    summary_df = pd.DataFrame(rows)
    return '\n'.join(lines), summary_df


# ---------------------------------------------------------------------------
# Table 3: Method comparison
# ---------------------------------------------------------------------------

def make_table3(ames_df, shap_df, pdp_df, dataset, config):
    decimals = get_decimals(config['outcome_type'])
    models = config['all_models']
    methods = ['AME', 'SHAP', 'PDP']
    n_data_cols = len(models) * len(methods)

    col_spec = 'l' + 'r' * n_data_cols
    caption = (
        f"{config['dataset_label']}: Method Comparison. "
        f"AME, SHAP, and PDP estimates for {config['outcome_label']}, "
        f"full specification (A+B+C)."
    )
    label = f"tab:{dataset}_method_comparison"

    lines = latex_begin(caption, label, col_spec)

    # Model header
    model_header = ''
    for model in models:
        model_header += f' & \\multicolumn{{{len(methods)}}}{{c}}{{{MODEL_LABELS[model]}}}'
    model_header += r' \\'
    lines.append(model_header)

    # Cmidrule
    cmidrule = ''
    col = 2
    for _ in models:
        cmidrule += f'\\cmidrule(lr){{{col}-{col + len(methods) - 1}}} '
        col += len(methods)
    lines.append(cmidrule)

    # Method header
    method_header = ''
    for _ in models:
        for method in methods:
            method_header += f' & {method}'
    method_header += r' \\'
    lines.append(method_header)
    lines.append(r'\midrule')

    full_ame = ames_df[ames_df['spec'] == 'ABC']
    available_features = full_ame['term'].unique().tolist()
    feature_groups = config['feature_groups']

    rows = []
    first_panel = True

    for panel_label, features in iter_feature_groups(feature_groups, available_features):
        if not first_panel:
            lines.append(r'\midrule')
        first_panel = False

        for feature in features:
            row = {'feature': feature}
            cells = []

            for model in models:
                # AME
                ame_sub = full_ame[
                    (full_ame['term'] == feature) &
                    (full_ame['model'] == model)
                ]
                ame_val = ame_sub['estimate'].values[0] if len(ame_sub) > 0 else np.nan
                row[f'{model}_ame'] = ame_val
                cells.append(fmt_point(ame_val, decimals))

                # SHAP
                shap_val = np.nan
                if shap_df is not None:
                    shap_sub = shap_df[
                        (shap_df['feature'] == feature) &
                        (shap_df['model'] == model)
                    ]
                    if len(shap_sub) > 0:
                        shap_val = shap_sub['shap_estimate'].values[0]
                row[f'{model}_shap'] = shap_val
                cells.append(fmt_point(shap_val, decimals))

                # PDP
                pdp_val = np.nan
                if pdp_df is not None:
                    pdp_sub = pdp_df[
                        (pdp_df['feature'] == feature) &
                        (pdp_df['model'] == model)
                    ]
                    if len(pdp_sub) > 0:
                        pdp_val = pdp_sub['pdp_estimate'].values[0]
                row[f'{model}_pdp'] = pdp_val
                cells.append(fmt_point(pdp_val, decimals))

            rows.append(row)

            data_row = escape_feature(feature)
            for cell in cells:
                data_row += f' & {cell}'
            data_row += r' \\'
            lines.append(data_row)

    lines.extend(latex_end_no_stars())
    summary_df = pd.DataFrame(rows)
    return '\n'.join(lines), summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Simulation results paths
# ---------------------------------------------------------------------------

SIM1_RESULTS_DIR = os.path.join(PAPER_DIR, 'simulations', 'sim1_ame_recovery', 'results')

SAMPLE_SIZES = [250, 500, 1000, 2500, 5000]

SIM_MODELS_REGRESSION = ['linear', 'rf', 'xgboost', 'tensorflow']
SIM_MODELS_CLASSIFICATION = ['logistic', 'rf', 'xgboost', 'tensorflow']

SIM_MODEL_LABELS = {
    'linear':     'Linear',
    'logistic':   'Logistic',
    'rf':         'Random Forest',
    'xgboost':    'XGBoost',
    'tensorflow': 'Neural Net',
}

# True AMEs for each DGP and feature
# regression: linear y = 2*x1 + 3*x2, nonlinear y = 2*x1^2 + 3*x2
# classification: same DGPs wrapped in sigmoid
TRUE_AMES = {
    'regression': {
        'linear':      {'x1': 2.0,  'x2': 3.0,  'x3': 0.0, 'x4': 0.0},
        'nonlinear':   {'x1': 0.0,  'x2': 3.0,  'x3': 0.0, 'x4': 0.0},
        'interaction': {'x1': 2.0,  'x2': 3.0,  'x3': 0.0, 'x4': 0.0},
    },
    'classification': {
        'linear':      {'x1': 0.199, 'x2': 0.298, 'x3': 0.0, 'x4': 0.0},
        'nonlinear':   {'x1': 0.0,   'x2': 0.298, 'x3': 0.0, 'x4': 0.0},
        'interaction': {'x1': 0.199, 'x2': 0.298, 'x3': 0.0, 'x4': 0.0},
    },
}

FEATURES = ['x1', 'x2', 'x3', 'x4']

DGP_LABELS = {
    'linear':      'Linear',
    'nonlinear':   'Nonlinear',
    'interaction': 'Interaction',
}

OUTCOME_TYPE_LABELS = {
    'regression':     'Regression',
    'classification': 'Classification',
}


# ---------------------------------------------------------------------------
# Simulation table helpers
# ---------------------------------------------------------------------------

def load_sim1_results(outcome_type: str, dgp: str) -> pd.DataFrame:
    """
    Load and concatenate all sim1 results for a given outcome type and DGP.

    Returns a DataFrame with columns:
        iteration, dgp, n, model, feature, ame_estimate, true_ame, ...
    """
    prefix = 'regression' if outcome_type == 'regression' else 'classification'
    models = SIM_MODELS_REGRESSION if outcome_type == 'regression' else SIM_MODELS_CLASSIFICATION
    dfs = []
    for model in models:
        for n in SAMPLE_SIZES:
            fname = f'{prefix}_{dgp}_n{n}_{model}.parquet'
            fpath = os.path.join(SIM1_RESULTS_DIR, fname)
            if os.path.exists(fpath):
                df = pd.read_parquet(fpath)
                df['model'] = model
                df['n'] = n
                df['dgp'] = dgp
                dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compute_bias_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean bias and RMSE per model, n, feature.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results with columns: model, n, feature, ame_estimate, true_ame

    Returns
    -------
    pd.DataFrame
        Columns: model, n, feature, bias, rmse
    """
    rows = []
    for (model, n, feature), group in df.groupby(['model', 'n', 'feature']):
        bias = (group['ame_estimate'] - group['true_ame']).mean()
        rmse = np.sqrt(((group['ame_estimate'] - group['true_ame']) ** 2).mean())
        rows.append({
            'model': model,
            'n': n,
            'feature': feature,
            'bias': bias,
            'rmse': rmse,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulation Table: AME Recovery (one per DGP x outcome_type)
# ---------------------------------------------------------------------------

def make_sim1_table(
    outcome_type: str,
    dgp: str,
) -> tuple:
    """
    Generate Simulation 1 AME Recovery table for one DGP and outcome type.

    Structure:
        Rows: features (x1, x2, x3, x4) with true AME shown
              grouped by model with panel labels
        Columns: n=250, n=500, n=1000, n=2500, n=5000
        Each cell: bias on top, (RMSE) below

    Parameters
    ----------
    outcome_type : str
        'regression' or 'classification'
    dgp : str
        'linear', 'nonlinear', or 'interaction'

    Returns
    -------
    tuple
        (latex_string, summary_df)
    """
    df = load_sim1_results(outcome_type, dgp)
    if df.empty:
        print(f"    WARNING: No data found for {outcome_type} {dgp}")
        return '', pd.DataFrame()

    stats = compute_bias_rmse(df)
    true_ames = TRUE_AMES[outcome_type][dgp]

    n_cols = len(SAMPLE_SIZES)
    col_spec = 'lr' + 'r' * n_cols  # feature, true AME, then sample size columns

    outcome_label = OUTCOME_TYPE_LABELS[outcome_type]
    dgp_label = DGP_LABELS[dgp]

    caption = (
        f"Simulation 1: AME Recovery — {dgp_label} DGP ({outcome_label}). "
        f"Mean bias and RMSE (in parentheses) across 1{',000' if outcome_type == 'regression' else ',000'} "
        f"Monte Carlo iterations."
    )
    label = f"tab:sim1_{outcome_type}_{dgp}"

    lines = latex_begin(caption, label, col_spec)

    # Spanning header: "Bias (RMSE)" over sample size columns
    span_header = f' & & \\multicolumn{{{len(SAMPLE_SIZES)}}}{{c}}{{Bias (RMSE)}} \\\\'
    lines.append(span_header)
    lines.append(f'\\cmidrule(lr){{3-{2 + len(SAMPLE_SIZES)}}}')

    # Column header row
    header = 'Variable & True AME'
    for n in SAMPLE_SIZES:
        header += f' & $n={n:,}$'
    header += r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    # Data rows — grouped by model
    summary_rows = []
    first_model = True

    models = SIM_MODELS_REGRESSION if outcome_type == 'regression' else SIM_MODELS_CLASSIFICATION
    for model in models:
        # Model panel label
        if not first_model:
            lines.append(r'\midrule')
        first_model = False
        lines.append(
            f"\\multicolumn{{{n_cols + 2}}}{{l}}"
            f"{{\\textit{{Panel: {SIM_MODEL_LABELS[model]}}}}} \\\\"
        )

        for feature in FEATURES:
            true_ame = true_ames.get(feature, 0.0)
            bias_cells = []
            rmse_cells = []

            for n in SAMPLE_SIZES:
                subset = stats[
                    (stats['model'] == model) &
                    (stats['n'] == n) &
                    (stats['feature'] == feature)
                ]
                if len(subset) == 0:
                    bias_cells.append('--')
                    rmse_cells.append('')
                else:
                    bias = subset['bias'].values[0]
                    rmse = subset['rmse'].values[0]
                    bias_cells.append(f"{bias:.4f}")
                    rmse_cells.append(f"({rmse:.4f})")

                    summary_rows.append({
                        'outcome_type': outcome_type,
                        'dgp': dgp,
                        'model': model,
                        'n': n,
                        'feature': feature,
                        'true_ame': true_ame,
                        'bias': bias,
                        'rmse': rmse,
                    })

            # Bias row
            bias_row = f'\\quad {escape_feature(feature)} & {true_ame:.3f}'
            for cell in bias_cells:
                bias_row += f' & {cell}'
            bias_row += r' \\'
            lines.append(bias_row)

            # RMSE row
            rmse_row = ' & '
            for cell in rmse_cells:
                rmse_row += f' & {cell}'
            rmse_row += r' \\'
            lines.append(rmse_row)

    # Footer — no stars for simulation tables
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item \textit{Notes:} Mean bias and RMSE (in parentheses) computed over '
        f'{"1,000" if outcome_type == "regression" else "500"} Monte Carlo iterations. '
        r'True AMEs computed via Monte Carlo integration with $n=1{,}000{,}000$ observations. '
        r'Noise features x3 and x4 have true AME of zero.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table*}',
    ]

    summary_df = pd.DataFrame(summary_rows)
    return '\n'.join(lines), summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)

    # --- Empirical tables ---
    for dataset, config in RESULTS.items():
        print(f"\nGenerating empirical tables for {config['dataset_label']}...")

        if not os.path.exists(config['ames_path']):
            print(f"  WARNING: AME results not found: {config['ames_path']}")
            continue
        ames_df = pd.read_parquet(config['ames_path'])

        shap_df = pd.read_parquet(config['shap_path']) if os.path.exists(config['shap_path']) else None
        pdp_df  = pd.read_parquet(config['pdp_path'])  if os.path.exists(config['pdp_path'])  else None

        if shap_df is None:
            print(f"  WARNING: SHAP results not found")
        if pdp_df is None:
            print(f"  WARNING: PDP results not found")

        for table_num, make_fn, args in [
            (1, make_table1, (ames_df, dataset, config)),
            (2, make_table2, (ames_df, dataset, config)),
            (3, make_table3, (ames_df, shap_df, pdp_df, dataset, config)),
        ]:
            print(f"  Generating Table {table_num}...")
            tex, df = make_fn(*args)

            tex_path = os.path.join(TABLES_DIR, f'{dataset}_table{table_num}.tex')
            pq_path  = os.path.join(TABLES_DIR, f'{dataset}_table{table_num}.parquet')

            with open(tex_path, 'w') as f:
                f.write(tex)
            df.to_parquet(pq_path, index=False)
            print(f"    Saved: {tex_path}")

    # --- Simulation 1 tables ---
    print(f"\nGenerating simulation tables...")

    for outcome_type in ['regression', 'classification']:
        for dgp in ['linear', 'nonlinear', 'interaction']:
            label = f"{outcome_type}_{dgp}"
            print(f"  Generating sim1_{label}...")
            tex, df = make_sim1_table(outcome_type, dgp)

            if tex:
                tex_path = os.path.join(TABLES_DIR, f'sim1_{label}.tex')
                pq_path  = os.path.join(TABLES_DIR, f'sim1_{label}.parquet')

                with open(tex_path, 'w') as f:
                    f.write(tex)
                if not df.empty:
                    df.to_parquet(pq_path, index=False)
                print(f"    Saved: {tex_path}")

    print(f"\nAll tables generated. Output in: {TABLES_DIR}")


if __name__ == '__main__':
    main()
