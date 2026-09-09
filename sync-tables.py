#!/usr/bin/env python3
"""Copy freshly generated table bodies into paper_long.tex.

generate_tables.py writes a complete table* environment per table into
tables/*.tex, but paper_long.tex carries its own wrapper around each one --
some tables are wrapped in \\resizebox, and all of them are indented to match
the surrounding source. Rewriting the whole environment would throw that away,
so this script replaces only the part that actually changes: everything from
\\begin{tabular} through \\end{tablenotes}.

Tables are matched by \\label, which both files carry, so the mapping does not
depend on file order or on the caption text.

Usage:  python sync-tables.py [--check]

--check reports what would change without writing.
"""
import os
import re
import sys

PAPER = 'paper_long.tex'
TABLES_DIR = 'tables'

# label -> generated file, for every table paper_long.tex embeds
TABLE_FILES = {
    'tab:adult_spec_search':                  'adult_table1.tex',
    'tab:adult_model_comparison':             'adult_table2.tex',
    'tab:adult_method_comparison':            'adult_table3.tex',
    'tab:credit_default_spec_search':         'credit_default_table1.tex',
    'tab:credit_default_model_comparison':    'credit_default_table2.tex',
    'tab:credit_default_method_comparison':   'credit_default_table3.tex',
    'tab:ames_housing_spec_search':           'ames_housing_table1.tex',
    'tab:ames_housing_model_comparison':      'ames_housing_table2.tex',
    'tab:ames_housing_method_comparison':     'ames_housing_table3.tex',
    'tab:window_settings':                    'window_settings.tex',

    'tab:sim1_classification_linear':         'sim1_classification_linear.tex',
    'tab:sim1_classification_nonlinear':      'sim1_classification_nonlinear.tex',
    'tab:sim1_classification_interaction':    'sim1_classification_interaction.tex',
    'tab:sim1_regression_linear':             'sim1_regression_linear.tex',
    'tab:sim1_regression_nonlinear':          'sim1_regression_nonlinear.tex',
    'tab:sim1_regression_interaction':        'sim1_regression_interaction.tex',

    'tab:sim2_calibration_linear':            'sim2_calibration_linear.tex',
    'tab:sim2_calibration_regression_linear': 'sim2_calibration_regression_linear.tex',

    'tab:sim3_bias_regression_linear':        'sim3_bias_regression_linear.tex',
    'tab:sim3_bias_classification_linear':    'sim3_bias_classification_linear.tex',
    'tab:sim3_coverage_regression_linear':    'sim3_coverage_regression_linear.tex',
    'tab:sim3_coverage_classification_linear': 'sim3_coverage_classification_linear.tex',

    'tab:sim3_bias_regression_linear_uniform':  'sim3_bias_regression_linear_uniform.tex',
    'tab:sim3_bias_classification_linear_uniform': 'sim3_bias_classification_linear_uniform.tex',
}

BODY = re.compile(
    r'\\begin\{tabular\}.*?\\end\{tablenotes\}',
    re.DOTALL,
)


def extract_body(path):
    """Pull the tabular-through-tablenotes span out of a generated table."""
    with open(path, encoding='utf-8', newline='') as fh:
        text = fh.read().replace('\r\n', '\n')
    m = BODY.search(text)
    if m is None:
        raise ValueError(f'no tabular/tablenotes body found in {path}')
    return m.group(0)


def indent_like(body, indent):
    """
    Re-indent a generated body to sit at the paper's indentation level.

    The paper indents the rows of a table one level deeper than the
    \\begin{tabular} that opens it, so everything after the first line gets an
    extra tab.
    """
    out = []
    for i, line in enumerate(body.split('\n')):
        stripped = line.strip()
        if not stripped:
            out.append('')
        elif i == 0:
            out.append(stripped)
        else:
            out.append(indent + '\t' + stripped)
    return '\n'.join(out)


def replace_one(paper, label, body):
    """
    Swap the body of the table carrying `label`.

    Returns (text, status), status one of 'updated', 'current', 'absent'.
    A label the paper does not carry is 'absent' rather than an error: the
    journal paper and the ICDM paper embed overlapping but different sets.
    """
    anchor = f'\\label{{{label}}}'
    at = paper.find(anchor)
    if at < 0:
        return paper, 'absent'

    m = BODY.search(paper, at)
    if m is None:
        raise ValueError(f'no body found after {label} in {PAPER}')

    # Indentation of the \begin{tabular} line the paper currently uses.
    line_start = paper.rfind('\n', 0, m.start()) + 1
    indent = paper[line_start:m.start()]
    if indent.strip():
        indent = ''

    new_body = indent_like(body, indent)
    if paper[m.start():m.end()] == new_body:
        return paper, 'current'
    return paper[:m.start()] + new_body + paper[m.end():], 'updated'


def main():
    # Both targets are settable so the ICDM camera-ready can be synced from
    # its own pipeline's tables:
    #     python sync-tables.py --paper paper_icdm.tex --tables tables-icdm
    global PAPER, TABLES_DIR
    check = '--check' in sys.argv
    if '--paper' in sys.argv:
        PAPER = sys.argv[sys.argv.index('--paper') + 1]
    if '--tables' in sys.argv:
        TABLES_DIR = sys.argv[sys.argv.index('--tables') + 1]
    print(f'{PAPER}  <-  {TABLES_DIR}/\n')

    # newline='' so the file's own line endings survive the round trip --
    # paper_long.tex is CRLF, and rewriting it as LF would make every line of
    # the next git diff look changed.
    with open(PAPER, encoding='utf-8', newline='') as fh:
        raw = fh.read()
    crlf = '\r\n' in raw
    paper = raw.replace('\r\n', '\n')

    changed, skipped, absent = [], [], []
    for label, filename in TABLE_FILES.items():
        path = os.path.join(TABLES_DIR, filename)
        if not os.path.exists(path):
            skipped.append(f'{label}: {path} missing')
            continue
        paper, status = replace_one(paper, label, extract_body(path))
        if status == 'updated':
            changed.append(f'{label}: updated')
        elif status == 'current':
            skipped.append(f'{label}: already current')
        else:
            absent.append(label)

    for line in changed + skipped:
        print(' ', line)
    if absent:
        print(f'\n  ({len(absent)} table(s) not embedded in {PAPER})')

    if check:
        print('\n--check: nothing written')
        return

    if crlf:
        paper = paper.replace('\n', '\r\n')
    with open(PAPER, 'w', encoding='utf-8', newline='') as fh:
        fh.write(paper)
    print(f'\n{len(changed)} table(s) written into {PAPER}')


if __name__ == '__main__':
    main()
