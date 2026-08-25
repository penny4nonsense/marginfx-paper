#!/usr/bin/env python3
"""Copy result numbers from the main paper into the supplement.

imsart forbids the xr package, so cross-document references are resolved by
injecting \newlabel definitions harvested from marginfx-main.aux directly into
marginfx-supp.tex, under the M- prefix. Run after every main-paper rebuild.
"""
import re, sys, os

BACK = ['lem:sobolev_existence','lem:sobolev_consistency','lem:bv_existence',
        'lem:bv_consistency','lem:hadamard','thm:neural_nets','thm:boosting',
        'thm:forests','cor:bootstrap_nn','cor:bootstrap_boost','cor:bootstrap_rf',
        'def:wellbehaved_sobolev']

if not os.path.exists('marginfx-main.aux'):
    sys.exit("marginfx-main.aux not found -- compile marginfx-main.tex first.")
aux = open('marginfx-main.aux').read()

lines, missing = [], []
for k in BACK:
    m = re.search(r'\\newlabel\{'+re.escape(k)+r'\}\{\{([^}]*)\}\{([^}]*)\}', aux)
    if m:
        # number, page; anchor fields left empty (link targets live in the main PDF)
        lines.append(r'\newlabel{M-%s}{{%s}{%s}{}{}{}}' % (k, m.group(1), m.group(2)))
    else:
        missing.append(k)

block = ('%%__MAINLABELS__  <- do not edit by hand; regenerate with sync-labels.py\n'
         '\\makeatletter\n' + '\n'.join(lines) + '\n\\makeatother\n')

supp = open('marginfx-supp.tex').read()
if '%%__MAINLABELS__' not in supp:
    sys.exit("marker %%__MAINLABELS__ missing from marginfx-supp.tex")
pat = re.compile(r'%%__MAINLABELS__.*?\\makeatother\n', re.S)
supp = pat.sub(lambda _: block, supp) if pat.search(supp) \
       else supp.replace('%%__MAINLABELS__\n', block, 1)
open('marginfx-supp.tex','w').write(supp)

print("synced %d labels" % len(lines))
if missing:
    print("MISSING from main .aux: " + ", ".join(missing))
