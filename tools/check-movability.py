#!/usr/bin/env python3
"""For each region of jrssb-main.tex proposed for the supplement, report the
labels it defines and everything that references them.

Moving a block that owns a label referenced from the main text leaves a
backwards reference ("see the supplement" in the middle of an argument), and
moving one referenced from the supplement as M-<label> breaks the xr link
unless the reference is flipped to a local one.  Neither shows up as a LaTeX
error until the reference resolves to the wrong place, so check first.

Usage: check-movability.py [start:end ...]   (1-indexed line ranges of main)
"""
import io
import re
import sys
from pathlib import Path

REPO = Path(r"C:\Users\jason\Dropbox\projects\marginfx-paper")
main_lines = io.open(REPO / "jrssb-main.tex", encoding="utf-8").read().split("\n")
main = "\n".join(main_lines)
supp = io.open(REPO / "jrssb-supp.tex", encoding="utf-8").read()

REGIONS = sys.argv[1:] or ["470:562", "1038:1166", "1136:1200"]


def refs_of(text):
    out = []
    for cmd in ("ref", "eqref", "autoref", "Cref", "cref"):
        out += re.findall(r"\\%s\{([^}]+)\}" % cmd, text)
    return [r.strip() for g in out for r in g.split(",")]


for spec in REGIONS:
    a, b = (int(x) for x in spec.split(":"))
    block = "\n".join(main_lines[a - 1:b])
    rest_of_main = "\n".join(main_lines[:a - 1] + main_lines[b:])
    labels = re.findall(r"\\label\{([^}]+)\}", block)

    print("=" * 78)
    head = re.search(r"\\(?:sub)?section\*?\{([^}]*)\}", block)
    print("lines %s  (%d lines)  %s" % (spec, b - a + 1,
                                        head.group(1) if head else ""))
    print("=" * 78)
    if not labels:
        print("  defines no labels -- free to move")
    for lab in labels:
        from_main = rest_of_main.count("\\ref{%s}" % lab) \
            + rest_of_main.count("\\eqref{%s}" % lab)
        from_supp = supp.count("{M-%s}" % lab)
        verdict = "FREE" if not from_main and not from_supp else (
            "BLOCKED by main" if from_main else "flip %d supp ref(s)" % from_supp)
        print("  %-28s main:%-3d supp:%-3d  %s"
              % (lab, from_main, from_supp, verdict))

    # references the block makes that would become cross-document if moved
    out = sorted(set(r for r in refs_of(block)
                     if not r.startswith(("S-", "M-"))))
    if out:
        print("  -- would need M- prefix once in the supplement:")
        print("     " + ", ".join(out))
