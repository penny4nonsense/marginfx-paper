# build-jrssb.ps1 -- compile the JRSS-B main paper + supplement.
#
# Unlike the AoS pair (build.ps1), this pair uses article.cls, which permits
# xr-hyper, so there is no sync-labels.py step: each document declares
# \externaldocument for the other and picks the numbers out of its .aux.
#
# That still makes ordering matter. Each document can only resolve references
# into the other once the other has an .aux, and a reference that changes a
# page or section number needs one more round to settle. Two full rounds is
# always enough here; the script does two and then checks.
#
# Usage:  .\build-jrssb.ps1

$ErrorActionPreference = "Stop"

$docs = @("jrssb-main", "jrssb-supp")

function Invoke-Pdflatex($file) {
    pdflatex -interaction=nonstopmode "$file.tex" | Out-Null
}

# Round 1: establish .aux files (cross-document refs will be ?? on this pass)
# and resolve the bibliographies.
foreach ($d in $docs) {
    Write-Host "--- pass 1: $d ---" -ForegroundColor Cyan
    Invoke-Pdflatex $d
    bibtex $d | Out-Null
}

# Round 2: now each .aux exists, so xr-hyper can read the other document.
# Two passes each so that numbers and page references settle.
foreach ($pass in 2, 3) {
    foreach ($d in $docs) {
        Write-Host "--- pass ${pass}: $d ---" -ForegroundColor Cyan
        Invoke-Pdflatex $d
    }
}

# Report anything still unresolved, and the page counts, which matter here:
# JRSS-B asks that the submission stay under about 30 double-spaced pages.
Write-Host ""
Write-Host "=== result ===" -ForegroundColor Green
foreach ($d in $docs) {
    $log = Get-Content "$d.log" -Raw
    $undef = ([regex]::Matches($log, "(Reference|Citation) ``[^']*' on page")).Count
    $pages = [regex]::Matches($log, "\((\d+) pages?")
    $n = if ($pages.Count) { $pages[$pages.Count - 1].Groups[1].Value } else { "?" }
    $colour = if ($undef -eq 0) { "Green" } else { "Yellow" }
    Write-Host ("  {0,-12} {1,4} pages   {2} unresolved references" -f $d, $n, $undef) -ForegroundColor $colour
}
