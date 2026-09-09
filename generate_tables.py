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
# Both are overridable so the ICDM camera-ready can be built from its own
# pipeline (empirical-icdm/, the bootstrap estimator) without disturbing the
# journal paper's tables, which come from the debiased estimator in empirical/.
TABLES_DIR = os.environ.get(
    'MARGINFX_TABLES_DIR', os.path.join(PAPER_DIR, 'tables')
)
EMPIRICAL_DIR = os.environ.get(
    'MARGINFX_EMPIRICAL_DIR', os.path.join(PAPER_DIR, 'empirical')
)

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
# Simulation constants — defined here so all functions can access them
# ---------------------------------------------------------------------------

SIM1_RESULTS_DIR = os.path.join(PAPER_DIR, 'simulations', 'sim1_ame_recovery', 'results')
SIM2_RESULTS_DIR = os.path.join(PAPER_DIR, 'simulations', 'sim2_se_calibration', 'results')

SAMPLE_SIZES = [250, 500, 1000, 2500, 5000]
CALIBRATION_SAMPLE_SIZES = [250, 1000, 5000]

SIM_MODELS_REGRESSION = ['linear', 'rf', 'xgboost', 'tensorflow']
SIM_MODELS_CLASSIFICATION = ['logistic', 'rf', 'xgboost', 'tensorflow']

# Classification calibration models
CALIBRATION_MODELS = ['logistic', 'rf', 'xgboost', 'tensorflow']

# Regression calibration models
CALIBRATION_MODELS_REGRESSION = ['linear', 'rf', 'xgboost', 'tensorflow']

SIM_MODEL_LABELS = {
    'linear':     'Linear',
    'logistic':   'Logistic',
    'rf':         'Random Forest',
    'xgboost':    'XGBoost',
    'tensorflow': 'Neural Net',
}

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
    # Same predictor as 'linear', drawn on bounded support. Unit variance per
    # coordinate, so the two are directly comparable.
    'linear_uniform': 'Bounded-Support Linear',
}

OUTCOME_TYPE_LABELS = {
    'regression':     'Regression',
    'classification': 'Classification',
}


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


# Which estimator produced the empirical tables. The two papers use different
# ones, and the table notes have to say which -- a debiased note on a bootstrap
# table misdescribes the method entirely. Set MARGINFX_METHOD=bootstrap when
# generating for the conference paper.
METHOD = os.environ.get('MARGINFX_METHOD', 'debiased')

_NOTE_STARS = {
    'debiased': (
        r'Estimates are the debiased cross-fitted window AME with $K=5$ folds; '
        r'standard errors from the orthogonal score in parentheses. '
        r'Step sizes $h_j$ and trimmed fractions are reported in '
        r'Table~\ref{tab:window_settings}.'
    ),
    'bootstrap': (
        r'Estimates are plug-in window AMEs with adaptive step sizes; '
        r'standard errors in parentheses are the standard deviation across 200 '
        r'nonparametric bootstrap replicates, each refit from a fresh '
        r'initialization on the resample.'
    ),
}


def latex_end_with_stars() -> list:
    return [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item \textit{Notes:} $^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.10. '
        + _NOTE_STARS[METHOD],
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
# Fit statistics helper
# ---------------------------------------------------------------------------

def fmt_fit_stat(value: float, stat_name: str) -> str:
    if pd.isna(value):
        return '--'
    if stat_name == 'rmse':
        return f"{value:,.1f}"
    return f"{value:.3f}"


def add_fit_stat_rows(lines, ames_df, models, specs, n_data_cols):
    if 'fit_stat1' not in ames_df.columns:
        return
    lines.append(r'\midrule')
    stat1_name = ames_df['fit_stat1_name'].iloc[0]
    stat2_name = ames_df['fit_stat2_name'].iloc[0]
    stat1_label = {'mcfadden_r2': 'McFadden $R^2$', 'r2': '$R^2$'}.get(stat1_name, stat1_name)
    stat2_label = {'accuracy': 'Accuracy', 'rmse': 'RMSE'}.get(stat2_name, stat2_name)

    # Stat 1 row
    stat1_row = stat1_label
    for model in models:
        for spec in specs:
            subset = ames_df[(ames_df['model'] == model) & (ames_df['spec'] == spec)]
            val = subset['fit_stat1'].iloc[0] if len(subset) > 0 else np.nan
            stat1_row += f' & {fmt_fit_stat(val, stat1_name)}'
    stat1_row += r' \\'
    lines.append(stat1_row)

    # Stat 2 row
    stat2_row = stat2_label
    for model in models:
        for spec in specs:
            subset = ames_df[(ames_df['model'] == model) & (ames_df['spec'] == spec)]
            val = subset['fit_stat2'].iloc[0] if len(subset) > 0 else np.nan
            stat2_row += f' & {fmt_fit_stat(val, stat2_name)}'
    stat2_row += r' \\'
    lines.append(stat2_row)


# ---------------------------------------------------------------------------
# Table 1: Specification search
# ---------------------------------------------------------------------------

def make_table1(ames_df, dataset, config):
    decimals = get_decimals(config['outcome_type'])
    models = config['spec_models']
    n_data_cols = len(models) * len(SPECS_ORDER)

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

    available_features = ames_df['term'].unique().tolist()
    feature_groups = config['feature_groups']

    rows = []
    first_panel = True

    for panel_label, features in iter_feature_groups(feature_groups, available_features):
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

            est_row = f'\\quad {escape_feature(feature)}'
            for cell in est_cells:
                est_row += f' & {cell}'
            est_row += r' \\'
            lines.append(est_row)

            se_row = ''
            for cell in se_cells:
                se_row += f' & {cell}'
            se_row += r' \\'
            lines.append(se_row)

    add_fit_stat_rows(lines, ames_df, models, SPECS_ORDER, n_data_cols)
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

    add_fit_stat_rows(lines, full_df, models, ['ABC'], n_data_cols)
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

    model_header = ''
    for model in models:
        model_header += f' & \\multicolumn{{{len(methods)}}}{{c}}{{{MODEL_LABELS[model]}}}'
    model_header += r' \\'
    lines.append(model_header)

    cmidrule = ''
    col = 2
    for _ in models:
        cmidrule += f'\\cmidrule(lr){{{col}-{col + len(methods) - 1}}} '
        col += len(methods)
    lines.append(cmidrule)

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
                ame_sub = full_ame[
                    (full_ame['term'] == feature) &
                    (full_ame['model'] == model)
                ]
                ame_val = ame_sub['estimate'].values[0] if len(ame_sub) > 0 else np.nan
                row[f'{model}_ame'] = ame_val
                cells.append(fmt_point(ame_val, decimals))

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
# Simulation 1: AME Recovery helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Window settings: step sizes and trimming
# ---------------------------------------------------------------------------

def make_window_settings_table() -> tuple:
    """
    Report the estimand settings for every empirical feature.

    h and the trimmed fraction define theta_h, so they belong in the paper
    rather than only in the code. Both are properties of the feature column
    alone, so they are identical across models and specifications; the full
    specification is read for each dataset.

    Returns
    -------
    tuple
        (latex string, tidy DataFrame).
    """
    lines = latex_begin(
        caption=(
            'Window settings for the empirical applications. Step size '
            '$h_j$ and the fraction of observations receiving trimming '
            'weight zero.'
        ),
        label='tab:window_settings',
        col_spec='llrr',
    )
    lines.append(r'Dataset & Feature & $h_j$ & Trimmed \\')
    lines.append(r'\midrule')

    rows = []
    first = True
    for dataset, config in RESULTS.items():
        if not os.path.exists(config['ames_path']):
            continue
        df = pd.read_parquet(config['ames_path'])
        df = df[df['spec'] == 'ABC']
        if df.empty:
            continue

        available = list(df['term'].unique())
        ordered = []
        for _, feats in iter_feature_groups(config['feature_groups'], available):
            ordered.extend(feats)

        if not first:
            lines.append(r'\midrule')
        first = False

        label = config['dataset_label']
        for i, feat in enumerate(ordered):
            sub = df[df['term'] == feat]
            if sub.empty:
                continue
            h = sub['h'].values[0] if 'h' in sub.columns else np.nan
            tr = sub['trimmed'].values[0] if 'trimmed' in sub.columns else np.nan

            if pd.isna(h):
                h_txt = '---'
            elif abs(h) >= 100:
                h_txt = f'{h:,.0f}'
            else:
                h_txt = f'{h:.4g}'
            tr_txt = '---' if pd.isna(tr) else f'{100 * tr:.1f}\\%'

            name = label if i == 0 else ''
            lines.append(
                f'{name} & {escape_feature(feat)} & {h_txt} & {tr_txt} \\\\'
            )
            rows.append({'dataset': dataset, 'term': feat, 'h': h,
                         'trimmed': tr})

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item \textit{Notes:} $h_j$ is the default adaptive step size, '
        r'$\max(10^{-4},\, 0.05\hat{\sigma}_j)$, floored at $0.5$ for '
        r'integer-valued features so that a count is contrasted over a whole '
        r'unit. Features declared binary use the level contrast '
        r'$f(x \mid x_j=1) - f(x \mid x_j=0)$, for which $h_j$ is undefined '
        r'and no trimming applies. The trimmed fraction is the share of '
        r'observations lying within $h_j$ of the boundary of the observed '
        r'support of $x_j$, which receive weight zero; for those features '
        r'the estimand is the window effect on the untrimmed subpopulation.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table*}',
    ]

    return '\n'.join(lines), pd.DataFrame(rows)


def load_sim1_results(outcome_type: str, dgp: str) -> pd.DataFrame:
    """Load and concatenate all sim1 results for a given outcome type and DGP."""
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
    """Compute mean bias and RMSE per model, n, feature."""
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


def make_sim1_table(outcome_type: str, dgp: str) -> tuple:
    """
    Generate Simulation 1 AME Recovery table for one DGP and outcome type.

    Rows: features grouped by model with panel labels
    Columns: n=250, n=500, n=1000, n=2500, n=5000
    Each cell: bias on top, (RMSE) below
    """
    df = load_sim1_results(outcome_type, dgp)
    if df.empty:
        print(f"    WARNING: No data found for {outcome_type} {dgp}")
        return '', pd.DataFrame()

    stats = compute_bias_rmse(df)
    true_ames = TRUE_AMES[outcome_type][dgp]

    n_cols = len(SAMPLE_SIZES)
    col_spec = 'lr' + 'r' * n_cols

    outcome_label = OUTCOME_TYPE_LABELS[outcome_type]
    dgp_label = DGP_LABELS[dgp]

    n_iters = '1,000' if outcome_type == 'regression' else '500'
    caption = (
        f"Simulation 1: AME Recovery --- {dgp_label} DGP ({outcome_label}). "
        f"Mean bias and RMSE (in parentheses) across {n_iters} Monte Carlo iterations."
    )
    label = f"tab:sim1_{outcome_type}_{dgp}"

    lines = latex_begin(caption, label, col_spec)

    span_header = f' & & \\multicolumn{{{n_cols}}}{{c}}{{Bias (RMSE)}} \\\\'
    lines.append(span_header)
    lines.append(f'\\cmidrule(lr){{3-{2 + n_cols}}}')

    header = 'Variable & True AME'
    for n in SAMPLE_SIZES:
        header += f' & $n={n:,}$'
    header += r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    summary_rows = []
    first_model = True

    models = SIM_MODELS_REGRESSION if outcome_type == 'regression' else SIM_MODELS_CLASSIFICATION
    for model in models:
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

            bias_row = f'\\quad {escape_feature(feature)} & {true_ame:.3f}'
            for cell in bias_cells:
                bias_row += f' & {cell}'
            bias_row += r' \\'
            lines.append(bias_row)

            rmse_row = ' & '
            for cell in rmse_cells:
                rmse_row += f' & {cell}'
            rmse_row += r' \\'
            lines.append(rmse_row)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item \textit{Notes:} Mean bias and RMSE (in parentheses) computed over '
        f'{n_iters} Monte Carlo iterations. '
        r'True AMEs computed via Monte Carlo integration with $n=1{,}000{,}000$ observations. '
        r'Noise features x3 and x4 have true AME of zero.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table*}',
    ]

    summary_df = pd.DataFrame(summary_rows)
    return '\n'.join(lines), summary_df


# ---------------------------------------------------------------------------
# Simulation 2: SE Calibration helpers (shared)
# ---------------------------------------------------------------------------

def compute_coverage_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean coverage and mean CI width per model, n, feature."""
    rows = []
    for (model, n, feature), group in df.groupby(['model', 'n', 'feature']):
        rows.append({
            'model': model,
            'n': n,
            'feature': feature,
            'coverage': group['covered'].mean(),
            'ci_width': group['ci_width'].mean(),
        })
    return pd.DataFrame(rows)


def _make_calibration_table(
    stats: pd.DataFrame,
    available_models: list,
    true_ame_map: dict,
    caption: str,
    label: str,
) -> tuple:
    """
    Shared table builder for classification and regression calibration tables.
    """
    n_cols = len(CALIBRATION_SAMPLE_SIZES)
    col_spec = 'lr' + 'r' * n_cols

    lines = latex_begin(caption, label, col_spec)

    span_header = f' & & \\multicolumn{{{n_cols}}}{{c}}{{Coverage (CI Width)}} \\\\'
    lines.append(span_header)
    lines.append(f'\\cmidrule(lr){{3-{2 + n_cols}}}')

    header = 'Variable & True AME'
    for n in CALIBRATION_SAMPLE_SIZES:
        header += f' & $n={n:,}$'
    header += r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    summary_rows = []
    first_model = True

    for model in available_models:
        if not first_model:
            lines.append(r'\midrule')
        first_model = False
        lines.append(
            f"\\multicolumn{{{n_cols + 2}}}{{l}}"
            f"{{\\textit{{Panel: {SIM_MODEL_LABELS[model]}}}}} \\\\"
        )

        for feature in FEATURES:
            true_ame = true_ame_map.get(feature, 0.0)
            cov_cells = []
            width_cells = []

            for n in CALIBRATION_SAMPLE_SIZES:
                subset = stats[
                    (stats['model'] == model) &
                    (stats['n'] == n) &
                    (stats['feature'] == feature)
                ]
                if len(subset) == 0:
                    cov_cells.append('--')
                    width_cells.append('')
                else:
                    coverage = subset['coverage'].values[0]
                    ci_width = subset['ci_width'].values[0]
                    cov_cells.append(f"{coverage:.3f}")
                    width_cells.append(f"({ci_width:.4f})")
                    summary_rows.append({
                        'model': model,
                        'n': n,
                        'feature': feature,
                        'true_ame': true_ame,
                        'coverage': coverage,
                        'ci_width': ci_width,
                    })

            cov_row = f'\\quad {escape_feature(feature)} & {true_ame:.3f}'
            for cell in cov_cells:
                cov_row += f' & {cell}'
            cov_row += r' \\'
            lines.append(cov_row)

            width_row = ' & '
            for cell in width_cells:
                width_row += f' & {cell}'
            width_row += r' \\'
            lines.append(width_row)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item \textit{Notes:} Coverage rate and mean 95\% CI width (in parentheses). '
        r'Intervals are bootstrap percentile intervals. Logistic, random forest '
        r'and XGBoost use 1{,}000 Monte Carlo iterations and 1{,}000 resamples; '
        r'the neural network uses 500 and 200, since every one of its resamples '
        r'is a fresh training run. '
        r'True AMEs computed via Monte Carlo integration with $n=1{,}000{,}000$ observations. '
        r'Noise features x3 and x4 have true AME of zero.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table*}',
    ]

    return '\n'.join(lines), pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Simulation 2: Classification calibration table
# ---------------------------------------------------------------------------

def load_sim2_results(dgp: str) -> pd.DataFrame:
    """
    Load classification calibration results.

    The two papers ran this simulation under different bootstrap semantics --
    the conference paper recomputes the adaptive step size on each resample,
    the journal paper holds it fixed -- so they have separate result sets and
    separate replication counts. METHOD selects between them.
    """
    if METHOD == 'bootstrap':
        results_dir = os.path.join(
            os.path.dirname(SIM2_RESULTS_DIR), 'results_icdm'
        )
        prefix = 'calibration_icdm_'
    else:
        results_dir, prefix = SIM2_RESULTS_DIR, 'calibration_'

    dfs = []
    for model in CALIBRATION_MODELS:
        for n in CALIBRATION_SAMPLE_SIZES:
            fname = f'{prefix}{dgp}_n{n}_{model}.parquet'
            fpath = os.path.join(results_dir, fname)
            if os.path.exists(fpath):
                df = pd.read_parquet(fpath)
                dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def make_sim2_table(dgp: str = 'linear') -> tuple:
    """Generate Simulation 2 SE Calibration table — classification."""
    df = load_sim2_results(dgp)
    if df.empty:
        print(f"    WARNING: No classification calibration data found for dgp={dgp}")
        return '', pd.DataFrame()

    stats = compute_coverage_stats(df)
    available_models = [m for m in CALIBRATION_MODELS if m in stats['model'].unique()]
    true_ame_map = df.groupby('feature')['true_ame'].first().to_dict()

    if METHOD == 'bootstrap':
        caption = (
            f"Simulation 2: Bootstrap SE Calibration --- {DGP_LABELS[dgp]} DGP "
            f"(Classification). Nominal coverage target: 95\\%. Coverage rate "
            f"and mean 95\\% CI width (in parentheses) across 500 Monte Carlo "
            f"iterations with 200 bootstrap resamples each."
        )
    else:
        caption = (
            f"Simulation 2: Bootstrap SE Calibration --- {DGP_LABELS[dgp]} DGP (Classification). "
            f"Nominal coverage target: 95\\%. "
            f"Coverage rate and mean 95\\% CI width (in parentheses). Monte Carlo "
            f"iterations and bootstrap resamples are 1{{,}}000 each for logistic, "
            f"random forest and XGBoost, and 500 and 200 for the neural network."
        )
    label = f"tab:sim2_calibration_{dgp}"

    return _make_calibration_table(stats, available_models, true_ame_map, caption, label)


# ---------------------------------------------------------------------------
# Simulation 2: Regression calibration table
# ---------------------------------------------------------------------------

def load_sim2_regression_results(dgp: str) -> pd.DataFrame:
    """Load regression calibration results."""
    dfs = []
    for model in CALIBRATION_MODELS_REGRESSION:
        for n in CALIBRATION_SAMPLE_SIZES:
            fname = f'calibration_regression_{dgp}_n{n}_{model}.parquet'
            fpath = os.path.join(SIM2_RESULTS_DIR, fname)
            if os.path.exists(fpath):
                df = pd.read_parquet(fpath)
                dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def make_sim2_regression_table(dgp: str = 'linear') -> tuple:
    """Generate Simulation 2 SE Calibration table — regression."""
    df = load_sim2_regression_results(dgp)
    if df.empty:
        print(f"    WARNING: No regression calibration data found for dgp={dgp}")
        return '', pd.DataFrame()

    stats = compute_coverage_stats(df)
    available_models = [m for m in CALIBRATION_MODELS_REGRESSION if m in stats['model'].unique()]
    true_ame_map = df.groupby('feature')['true_ame'].first().to_dict()

    caption = (
        f"Simulation 2: Bootstrap SE Calibration --- {DGP_LABELS[dgp]} DGP (Regression). "
        f"Nominal coverage target: 95\\%. "
        f"Coverage rate and mean 95\\% CI width (in parentheses). Monte Carlo "
        f"iterations and bootstrap resamples are 1{{,}}000 each for logistic, "
        f"random forest and XGBoost, and 500 and 200 for the neural network."
    )
    label = f"tab:sim2_calibration_regression_{dgp}"

    return _make_calibration_table(stats, available_models, true_ame_map, caption, label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Simulation 3: debiased estimator with a closed-form representer
# ---------------------------------------------------------------------------

SIM3_RESULTS_DIR = os.path.join(PAPER_DIR, 'simulations', 'sim3_debiased', 'results')


def load_sim3_results(outcome_type: str, dgp: str) -> pd.DataFrame:
    """
    Load every Simulation 3 result file for one outcome type and DGP.

    Simulation 3 writes one file per model and sample size, each carrying
    both the point-estimate columns and the interval columns, so bias, RMSE
    and coverage all come from the same frame.
    """
    models = (SIM_MODELS_REGRESSION if outcome_type == 'regression'
              else SIM_MODELS_CLASSIFICATION)
    # Each arm writes to its own directory so they can coexist. The bounded
    # design lives in the uniform arm; everything else in the closed-form one.
    results_dir = SIM3_RESULTS_DIR
    if dgp.endswith('_uniform'):
        results_dir = os.path.join(
            os.path.dirname(SIM3_RESULTS_DIR), 'results_uniform'
        )

    dfs = []
    for model in models:
        for n in SAMPLE_SIZES:
            fname = f'{outcome_type}_{dgp}_n{n}_{model}.parquet'
            fpath = os.path.join(results_dir, fname)
            if os.path.exists(fpath):
                dfs.append(pd.read_parquet(fpath))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def make_sim3_bias_rmse_table(outcome_type: str, dgp: str) -> tuple:
    """
    Bias and RMSE of the debiased estimator, laid out like the Simulation 1
    table so the two can be read side by side.

    Returns
    -------
    tuple
        (latex string, tidy DataFrame). Both empty if no results exist yet.
    """
    df = load_sim3_results(outcome_type, dgp)
    if df.empty:
        return '', pd.DataFrame()

    stats = compute_bias_rmse(df)
    models = [m for m in (SIM_MODELS_REGRESSION if outcome_type == 'regression'
                          else SIM_MODELS_CLASSIFICATION)
              if m in set(stats['model'])]
    # Prefer the true values carried in the results themselves. The hardcoded
    # map holds only for designs where trimming is inactive; under an active
    # trimming weight the target depends on h and the support, so a constant
    # would drift away from what the run actually targeted. On the bounded
    # design the true effect of x1 is 1.943 rather than 2, for instance.
    if 'true_ame' in df.columns:
        true_map = (df.groupby('feature')['true_ame'].first().to_dict())
    else:
        true_map = TRUE_AMES[outcome_type][dgp]
    decimals = 4 if outcome_type == 'classification' else 3

    n_cols = len(SAMPLE_SIZES)
    lines = latex_begin(
        caption=(
            f'Simulation 3: debiased estimator, '
            f'{DGP_LABELS[dgp].lower()} '
            f'{OUTCOME_TYPE_LABELS[outcome_type].lower()} DGP. '
            f'Bias and RMSE (in parentheses) by sample size.'
        ),
        label=f'tab:sim3_bias_{outcome_type}_{dgp}',
        col_spec='lr' + 'r' * n_cols,
    )
    lines.append(f' & & \\multicolumn{{{n_cols}}}{{c}}{{Bias (RMSE)}} \\\\')
    lines.append(f'\\cmidrule(lr){{3-{2 + n_cols}}}')

    header = 'Variable & True AME'
    for n in SAMPLE_SIZES:
        header += f' & $n={n:,}$'
    header += r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    rows = []
    first = True
    for model in models:
        if not first:
            lines.append(r'\midrule')
        first = False
        lines.append(
            f"\\multicolumn{{{n_cols + 2}}}{{l}}"
            f"{{\\textit{{Panel: {SIM_MODEL_LABELS[model]}}}}} \\\\"
        )
        for feature in FEATURES:
            true_ame = true_map.get(feature, 0.0)
            bias_cells, rmse_cells = [], []
            for n in SAMPLE_SIZES:
                sub = stats[(stats['model'] == model) &
                            (stats['n'] == n) &
                            (stats['feature'] == feature)]
                if sub.empty:
                    bias_cells.append('--')
                    rmse_cells.append('')
                else:
                    b = float(sub['bias'].values[0])
                    r = float(sub['rmse'].values[0])
                    bias_cells.append(f'{b:.{decimals}f}')
                    rmse_cells.append(f'({r:.{decimals}f})')
                    rows.append({
                        'outcome_type': outcome_type, 'dgp': dgp,
                        'model': model, 'n': n, 'feature': feature,
                        'true_ame': true_ame, 'bias': b, 'rmse': r,
                    })
            lines.append(
                f'\\quad {escape_feature(feature)} & '
                f'{true_ame:.{decimals}f} & ' + ' & '.join(bias_cells) + r' \\'
            )
            lines.append(' &  & ' + ' & '.join(rmse_cells) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item \textit{Notes:} Debiased cross-fitted estimator with $K=5$ '
        r'folds and the closed-form Riesz representer '
        r'$\alpha_h(u) = e^{-h^2/2}\sinh(h u_j)/h$, exact for independent '
        r'standard normal covariates. Trimming is inactive because the '
        r'support is unbounded. Bias is the mean of '
        r'$\hat{\theta}_{j,h} - \theta_{j,h}$ and RMSE its root mean square, '
        r'over the Monte Carlo iterations. True values computed by Monte '
        r'Carlo integration at $n=1{,}000{,}000$. Noise features $x_3$ and '
        r'$x_4$ have a true effect of zero.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table*}',
    ]
    return '\n'.join(lines), pd.DataFrame(rows)


def make_sim3_coverage_comparison(outcome_type: str, dgp: str = 'linear') -> tuple:
    """
    The comparison the correction is for: coverage of the refitting-bootstrap
    intervals against coverage of the influence-function intervals, over the
    same DGP, learners and sample sizes.

    Simulation 2 supplies the bootstrap column and Simulation 3 the
    influence-function column. Only the sample sizes Simulation 2 was run at
    can appear, since the bootstrap side does not exist elsewhere.

    Returns
    -------
    tuple
        (latex string, tidy DataFrame). Both empty if either side is missing.
    """
    boot = (load_sim2_regression_results(dgp) if outcome_type == 'regression'
            else load_sim2_results(dgp))
    deb = load_sim3_results(outcome_type, dgp)
    if boot.empty or deb.empty:
        return '', pd.DataFrame()

    boot_stats = compute_coverage_stats(boot)
    deb_stats = compute_coverage_stats(deb)

    sizes = [n for n in CALIBRATION_SAMPLE_SIZES
             if n in set(boot_stats['n']) and n in set(deb_stats['n'])]
    if not sizes:
        return '', pd.DataFrame()

    models = (CALIBRATION_MODELS_REGRESSION if outcome_type == 'regression'
              else CALIBRATION_MODELS)
    models = [m for m in models
              if m in set(boot_stats['model']) and m in set(deb_stats['model'])]
    # Prefer the true values carried in the results themselves. The hardcoded
    # map holds only for designs where trimming is inactive; under an active
    # trimming weight the target depends on h and the support, so a constant
    # would drift away from what the run actually targeted. On the bounded
    # design the true effect of x1 is 1.943 rather than 2, for instance.
    if 'true_ame' in deb.columns:
        true_map = (deb.groupby('feature')['true_ame'].first().to_dict())
    else:
        true_map = TRUE_AMES[outcome_type][dgp]

    k = len(sizes)
    lines = latex_begin(
        caption=(
            f'Coverage of nominal 95\\% confidence intervals, '
            f'{DGP_LABELS[dgp].lower()} '
            f'{OUTCOME_TYPE_LABELS[outcome_type].lower()} DGP: '
            f'refitting bootstrap against the influence function.'
        ),
        label=f'tab:sim3_coverage_{outcome_type}_{dgp}',
        col_spec='lr' + 'r' * (2 * k),
    )
    lines.append(
        f' & & \\multicolumn{{{k}}}{{c}}{{Refitting bootstrap}} '
        f'& \\multicolumn{{{k}}}{{c}}{{Influence function}} \\\\'
    )
    lines.append(
        f'\\cmidrule(lr){{3-{2 + k}}} \\cmidrule(lr){{{3 + k}-{2 + 2 * k}}}'
    )

    header = 'Variable & True AME'
    for _ in range(2):
        for n in sizes:
            header += f' & $n={n:,}$'
    header += r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    rows = []
    first = True
    for model in models:
        if not first:
            lines.append(r'\midrule')
        first = False
        lines.append(
            f"\\multicolumn{{{2 * k + 2}}}{{l}}"
            f"{{\\textit{{Panel: {SIM_MODEL_LABELS[model]}}}}} \\\\"
        )
        for feature in FEATURES:
            true_ame = true_map.get(feature, 0.0)
            cells = []
            for stats in (boot_stats, deb_stats):
                for n in sizes:
                    sub = stats[(stats['model'] == model) &
                                (stats['n'] == n) &
                                (stats['feature'] == feature)]
                    cells.append('--' if sub.empty
                                 else f"{float(sub['coverage'].values[0]):.3f}")
            for i, n in enumerate(sizes):
                rows.append({
                    'outcome_type': outcome_type, 'dgp': dgp, 'model': model,
                    'n': n, 'feature': feature, 'true_ame': true_ame,
                    'coverage_bootstrap': cells[i],
                    'coverage_influence': cells[k + i],
                })
            lines.append(
                f'\\quad {escape_feature(feature)} & {true_ame:.3f} & '
                + ' & '.join(cells) + r' \\'
            )

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\begin{tablenotes}',
        r'\footnotesize',
        r'\item \textit{Notes:} Empirical coverage of nominal 95\% intervals. '
        r'The bootstrap columns resample and refit the learner, centring on '
        r'the plug-in average; the influence-function columns use the '
        r'debiased cross-fitted estimator with $K=5$ folds and the '
        r'closed-form representer. Both are evaluated against the same '
        r'window estimand $\theta_{j,h}$ and the same Monte Carlo ground '
        r'truth. Noise features $x_3$ and $x_4$ have a true effect of zero.',
        r'\end{tablenotes}',
        r'\end{threeparttable}',
        r'\end{table*}',
    ]
    return '\n'.join(lines), pd.DataFrame(rows)


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

    # --- Window settings (h and trimming), all datasets ---
    print("\nGenerating window settings table...")
    tex, df = make_window_settings_table()
    if tex:
        tex_path = os.path.join(TABLES_DIR, 'window_settings.tex')
        with open(tex_path, 'w') as f:
            f.write(tex)
        if not df.empty:
            df.to_parquet(
                os.path.join(TABLES_DIR, 'window_settings.parquet'),
                index=False,
            )
        print(f"    Saved: {tex_path}")

    # --- Simulation 1 tables ---
    print(f"\nGenerating simulation 1 tables...")

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

    # --- Simulation 2 tables ---
    print(f"\nGenerating simulation 2 calibration tables...")

    for dgp in ['linear']:
        # Classification calibration
        print(f"  Generating sim2_calibration_{dgp} (classification)...")
        tex, df = make_sim2_table(dgp)
        if tex:
            tex_path = os.path.join(TABLES_DIR, f'sim2_calibration_{dgp}.tex')
            pq_path  = os.path.join(TABLES_DIR, f'sim2_calibration_{dgp}.parquet')
            with open(tex_path, 'w') as f:
                f.write(tex)
            if not df.empty:
                df.to_parquet(pq_path, index=False)
            print(f"    Saved: {tex_path}")

        # Regression calibration
        print(f"  Generating sim2_calibration_regression_{dgp} (regression)...")
        tex, df = make_sim2_regression_table(dgp)
        if tex:
            tex_path = os.path.join(TABLES_DIR, f'sim2_calibration_regression_{dgp}.tex')
            pq_path  = os.path.join(TABLES_DIR, f'sim2_calibration_regression_{dgp}.parquet')
            with open(tex_path, 'w') as f:
                f.write(tex)
            if not df.empty:
                df.to_parquet(pq_path, index=False)
            print(f"    Saved: {tex_path}")

    # --- Simulation 3 tables (debiased estimator) ---
    print("\nGenerating simulation 3 tables...")

    for outcome_type in ['regression', 'classification']:
        for dgp in ['linear', 'nonlinear', 'interaction',
                    'linear_uniform']:
            label = f'{outcome_type}_{dgp}'
            tex, df = make_sim3_bias_rmse_table(outcome_type, dgp)
            if tex:
                tex_path = os.path.join(TABLES_DIR, f'sim3_bias_{label}.tex')
                with open(tex_path, 'w') as f:
                    f.write(tex)
                if not df.empty:
                    df.to_parquet(
                        os.path.join(TABLES_DIR, f'sim3_bias_{label}.parquet'),
                        index=False,
                    )
                print(f"    Saved: {tex_path}")
            else:
                print(f"  Skipping sim3_bias_{label} (no results yet)")

    for outcome_type in ['regression', 'classification']:
        tex, df = make_sim3_coverage_comparison(outcome_type, 'linear')
        if tex:
            name = f'sim3_coverage_{outcome_type}_linear'
            tex_path = os.path.join(TABLES_DIR, f'{name}.tex')
            with open(tex_path, 'w') as f:
                f.write(tex)
            if not df.empty:
                df.to_parquet(
                    os.path.join(TABLES_DIR, f'{name}.parquet'), index=False
                )
            print(f"    Saved: {tex_path}")
        else:
            print(f"  Skipping sim3 coverage comparison for {outcome_type} "
                  f"(needs both sim2 and sim3 results)")

    print(f"\nAll tables generated. Output in: {TABLES_DIR}")


if __name__ == '__main__':
    main()
