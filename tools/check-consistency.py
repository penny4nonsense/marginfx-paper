#!/usr/bin/env python3
"""Final consistency sweep over the JRSS-B pair.

Mechanical checks only -- the things that are easy to get wrong across a 46 +
99 page split and tedious to verify by eye:

  1. JRSS-B submission requirements: abstract word count, keyword count.
  2. Oxford spelling: no -ise/-isation forms (OUP retains -ize).
  3. Cross-document refs: every \\ref{M-...} in the supplement resolves to a
     label in the main text, and vice versa.
  4. Internal refs: every \\ref / \\eqref resolves to a label in the same file.
  5. Labels defined but never referenced (dead weight, not an error).
  6. Hypothesis/conclusion exponent mismatch: any theorem concluding
     well-behavedness in the Sobolev sense whose stated consistency
     hypothesis is in a norm other than L^p.
  7. Stale deterministic-boundedness language on random Sobolev norms.
"""
import io
import re
import sys
from pathlib import Path

REPO = Path(r"C:\Users\jason\Dropbox\projects\marginfx-paper")
MAIN = REPO / "jrssb-main.tex"
SUPP = REPO / "jrssb-supp.tex"

main = io.open(MAIN, encoding="utf-8").read()
supp = io.open(SUPP, encoding="utf-8").read()

problems = []
notes = []


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


main_c, supp_c = strip_comments(main), strip_comments(supp)


# ---- 1. abstract and keywords -------------------------------------------
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_c, re.S)
if not m:
    problems.append("no abstract found in main")
else:
    body = re.sub(r"\\[a-zA-Z]+\s*", " ", m.group(1))
    body = re.sub(r"[{}$\\]", " ", body)
    nw = len(body.split())
    (notes if nw <= 200 else problems).append(
        "abstract is %d words (JRSS-B limit 200)" % nw)

kw = re.search(r"[Kk]eywords?:?\}?\s*(.*?)\.\s*\n", main_c, re.S)
if kw:
    flat = " ".join(kw.group(1).split())
    items = [k.strip() for k in re.split(r"[;,]", flat) if k.strip()]
    items = [re.sub(r"[{}$\\]|textbf|emph", "", k).strip() for k in items]
    items = [k for k in items if k]
    lowered = [k.lower() for k in items]
    ok = 5 <= len(items) <= 6 and lowered == sorted(lowered)
    (notes if ok else problems).append(
        "keywords: %d found, alphabetical=%s -> %s"
        % (len(items), lowered == sorted(lowered), "; ".join(items)))
else:
    problems.append("could not locate a keyword list in main")


# ---- 2. Oxford spelling --------------------------------------------------
#   Oxford spelling retains -ize for verbs from Greek -izein, so normalise,
#   minimise, realise and friends are wrong.  It does NOT touch words that
#   were never -izein verbs: analyse (back-formed from analysis) and centre
#   are correct British forms and must not be flagged.
BAD = (r"\b\w*(?:normalis|minimis|maximis|parametris|generalis|characteris"
       r"|organis|recognis|summaris|regularis|penalis|discretis|stabilis"
       r"|optimis|utilis|standardis|realis|emphasis(?:e|ed|ing))\w*")
# "characteristic(s)" is correct, and "emphasis" the noun is too.
ALLOW = re.compile(r"^(characteristic|characteristics|characteristically|"
                   r"emphasis)$", re.I)
for name, text in (("main", main_c), ("supp", supp_c)):
    hits = sorted({w for w in re.findall(BAD, text, re.I)
                   if not ALLOW.match(w)})
    if hits:
        problems.append("%s: non-Oxford spellings %s" % (name, hits))
    else:
        notes.append("%s: no -ise/-isation forms" % name)


# ---- 3 & 4. references --------------------------------------------------
def labels(text):
    return set(re.findall(r"\\label\{([^}]+)\}", text))


def refs(text):
    out = []
    for cmd in ("ref", "eqref", "autoref", "Cref", "cref"):
        out += re.findall(r"\\%s\{([^}]+)\}" % cmd, text)
    return [r for group in out for r in group.split(",")]


main_lab, supp_lab = labels(main_c), labels(supp_c)
main_ref, supp_ref = refs(main_c), refs(supp_c)

for name, rs, own, other, prefix in (
        ("supp", supp_ref, supp_lab, main_lab, "M-"),
        ("main", main_ref, main_lab, supp_lab, "S-")):
    bad_x, bad_i = [], []
    for r in rs:
        r = r.strip()
        if r.startswith(prefix):
            if r[len(prefix):] not in other:
                bad_x.append(r)
        elif r.startswith(("M-", "S-")):
            bad_x.append(r)          # wrong prefix for this document
        elif r not in own:
            bad_i.append(r)
    if bad_x:
        problems.append("%s: cross-doc refs with no target: %s"
                        % (name, sorted(set(bad_x))))
    if bad_i:
        problems.append("%s: internal refs with no label: %s"
                        % (name, sorted(set(bad_i))))
    if not bad_x and not bad_i:
        notes.append("%s: all %d refs resolve" % (name, len(rs)))


# ---- 5. unreferenced labels --------------------------------------------
used = set(r.strip() for r in main_ref)
used |= set(r.strip()[2:] for r in supp_ref if r.strip().startswith("M-"))
dead_main = sorted(l for l in main_lab if l not in used
                   and not l.startswith(("sec:", "app:", "fig:")))
used2 = set(r.strip() for r in supp_ref)
used2 |= set(r.strip()[2:] for r in main_ref if r.strip().startswith("S-"))
dead_supp = sorted(l for l in supp_lab if l not in used2
                   and not l.startswith(("sec:", "app:", "fig:")))
if dead_main:
    notes.append("main: labels never referenced: %s" % dead_main)
if dead_supp:
    notes.append("supp: labels never referenced: %s" % dead_supp)


# ---- 6. Sobolev hypothesis/conclusion exponent -------------------------
for m in re.finditer(r"\\begin\{theorem\}(.*?)\\end\{theorem\}", supp_c, re.S):
    blk = m.group(1)
    lab = re.search(r"\\label\{([^}]+)\}", blk)
    lab = lab.group(1) if lab else "?"
    if "Sobolev sense" not in blk:
        continue
    hyp = re.findall(r"\\norm\{\\fn - f\}_\{L\^?\{?(\w+)\}?\}", blk)
    if not hyp:
        problems.append("%s: concludes Sobolev well-behavedness, no "
                        "consistency hypothesis found" % lab)
    elif any(h != "p" for h in hyp):
        problems.append("%s: Sobolev conclusion but consistency hypothesis "
                        "in L^%s (Definition requires L^p)" % (lab, hyp))
    else:
        notes.append("%s: consistency hypothesis is L^p, matches definition"
                     % lab)


# ---- 7. stale deterministic boundedness --------------------------------
STALE = re.compile(r"\\sup_n\s*\\norm\{\\fn\}_\{?\\?W?\w*\{?2,p\}?[^<]*<\s*\\infty")
for name, text in (("main", main_c), ("supp", supp_c)):
    if STALE.search(text):
        problems.append("%s: still asserts sup_n ||f_n||_{W^{2,p}} < infinity"
                        % name)
if not any("sup_n" in p for p in problems):
    notes.append("no deterministic sup_n bound on a random Sobolev norm")


# ---- report -------------------------------------------------------------
print("=== notes ===")
for n in notes:
    print("  " + n)
print()
if problems:
    print("=== PROBLEMS ===")
    for p in problems:
        print("  " + p)
    sys.exit(1)
print("=== no problems found ===")
