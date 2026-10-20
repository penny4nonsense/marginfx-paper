# build.ps1 -- compile marginfx main paper + supplement in the right order.
#
# The supplement cites results by number from the main paper. imsart forbids
# the xr package, so sync-labels.py copies those numbers out of the main
# paper's .aux and injects them into the supplement. That makes ordering
# matter: main -> sync -> supplement -> main.
#
# Usage:  .\build.ps1

$ErrorActionPreference = "Stop"

function Invoke-Tex($file) {
    Write-Host "--- pdflatex/bibtex: $file ---" -ForegroundColor Cyan
    pdflatex -interaction=nonstopmode $file | Out-Null
    bibtex   $file | Out-Null
    pdflatex -interaction=nonstopmode $file | Out-Null
    pdflatex -interaction=nonstopmode $file | Out-Null
}

# 1. Main paper -- establishes theorem/lemma numbers.
Invoke-Tex "marginfx-main"

# 2. Copy those numbers into the supplement source.
Write-Host "--- syncing cross-document labels ---" -ForegroundColor Cyan
python sync-labels.py
if ($LASTEXITCODE -ne 0) { throw "sync-labels.py failed" }

# 3. Supplement.
Invoke-Tex "marginfx-supp"

# 4. Main again, so its page numbers settle.
Invoke-Tex "marginfx-main"

Write-Host ""
Write-Host "Done. Checking for unresolved references..." -ForegroundColor Green
foreach ($f in @("marginfx-main.log","marginfx-supp.log")) {
    $bad = Select-String -Path $f -Pattern "Reference ``|Citation ``" -ErrorAction SilentlyContinue
    if ($bad) {
        Write-Host "  $f :" -ForegroundColor Yellow
        $bad | ForEach-Object { Write-Host "    $($_.Line.Trim())" }
    } else {
        Write-Host "  $f : clean" -ForegroundColor Green
    }
}
