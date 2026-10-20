#!/usr/bin/env python3
"""Where the pages are in jrssb-main.pdf, section by section.

Walks the PDF a page at a time, matches numbered section and subsection
headings against the headings declared in jrssb-main.tex, and reports how many
pages each one spans.  Cutting decisions need this: the eye is a poor judge of
which section is long, and double spacing hides it further.
"""
import io
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\jason\Dropbox\projects\marginfx-paper")
TEX = REPO / "jrssb-main.tex"
PDF = REPO / "jrssb-main.pdf"

tex = io.open(TEX, encoding="utf-8").read()
tex = re.sub(r"(?<!\\)%.*", "", tex)


def clean(t):
    t = re.sub(r"\\[a-zA-Z]+\s*", " ", t)
    t = re.sub(r"[{}$\\~]", " ", t)
    return " ".join(t.split())


# declared headings, in document order, with the numbers LaTeX will print.
# Starred headings (Data availability, Acknowledgements) carry no number, and
# their names are distinctive enough to match on text alone.
heads = []
sec = sub = 0
for m in re.finditer(r"\\(section|subsection)(\*?)\{((?:[^{}]|\{[^{}]*\})*)\}", tex):
    kind, star, name = m.group(1), m.group(2), clean(m.group(3))
    if star:
        heads.append((kind, name, None))
        continue
    if kind == "section":
        sec += 1
        sub = 0
        heads.append((kind, name, "%d" % sec))
    else:
        sub += 1
        heads.append((kind, name, "%d.%d" % (sec, sub)))

npages = int(re.search(r"/Count\s+(\d+)",
                       io.open(PDF, "rb").read().decode("latin-1")).group(1)) \
    if False else None

# page count from pdfinfo if available, else probe
try:
    out = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True)
    npages = int(re.search(r"Pages:\s+(\d+)", out.stdout).group(1))
except Exception:
    sys.exit("pdfinfo unavailable")

pages = []
for p in range(1, npages + 1):
    r = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), "-layout",
                        str(PDF), "-"], capture_output=True)
    pages.append(r.stdout.decode("utf-8", errors="replace"))

# The page whose text contains the heading as LaTeX prints it: the number,
# then the title, at the start of a line.  Anchoring on the number is what
# stops "Method" matching the word "method" in the introduction.
ordered = []
cursor = 1
for kind, name, num in heads:
    if not name:
        continue
    words = name.split()[:4]
    body = r"\s+".join(re.escape(w) for w in words)
    pat = re.compile(r"^\s*" + (re.escape(num) + r"\s+" if num else "") + body,
                     re.M | re.I)
    hit = None
    for p in range(cursor, npages + 1):
        if pat.search(pages[p - 1]):
            hit = p
            cursor = p
            break
    if hit is None:
        print("  !! no page found for %s %s" % (num or "*", name))
        continue
    ordered.append((kind, ("%s %s" % (num, name)) if num else name, hit))

print("jrssb-main.pdf: %d pages\n" % npages)
print("%-6s %-62s %6s %6s" % ("level", "heading", "start", "pages"))
print("-" * 84)
for i, (kind, name, p) in enumerate(ordered):
    nxt = ordered[i + 1][2] if i + 1 < len(ordered) else npages + 1
    span = max(nxt - p, 0)
    lead = "  " if kind == "subsection" else ""
    flag = "  <<<" if kind == "section" and span >= 6 else ""
    print("%-6s %-62s %6d %6d%s"
          % (kind[:4], (lead + name)[:62], p, span, flag))

print()
print("top-level totals")
tops = [(n, p) for (k, n, p) in ordered if k == "section"]
for i, (n, p) in enumerate(tops):
    nxt = tops[i + 1][1] if i + 1 < len(tops) else npages + 1
    print("  %-56s %3d pages" % (n[:56], nxt - p))
