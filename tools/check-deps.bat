@echo off
REM check-deps.bat -- report which LaTeX packages imsart and this paper need
REM but your TeX installation is missing. Uses kpsewhich, which ships with
REM every TeX Live install.
REM
REM Usage:  check-deps.bat

setlocal enabledelayedexpansion

set MISSING=
set PKGS=amsmath amssymb amsthm bm color courier enumitem etoolbox fontenc ^
graphicx helvet hyperref keyval letterspace lmodern mathptmx mathrsfs natbib ^
newtxmath rotating textcase textcomp times mathtools booktabs multirow ^
threeparttable tabularx enumerate

echo Checking LaTeX dependencies...
echo.

for %%P in (%PKGS%) do (
    kpsewhich %%P.sty >nul 2>&1
    if errorlevel 1 (
        echo   MISSING  %%P.sty
        set MISSING=!MISSING! %%P
    )
)

echo.
if "!MISSING!"=="" (
    echo All dependencies present.
) else (
    echo ---------------------------------------------------------------
    echo Missing:!MISSING!
    echo.
    echo To install, run an ADMINISTRATOR command prompt and use:
    echo.
    echo     tlmgr install!MISSING!
    echo.
    echo If you do not have administrator rights, use TeX Live user mode:
    echo.
    echo     tlmgr init-usertree
    echo     tlmgr --usermode install!MISSING!
    echo.
    echo Note: some names above are provided by a larger bundle --
    echo   letterspace  is in  microtype
    echo   newtxmath    is in  newtx
    echo   mathrsfs     is in  jknapltx
    echo   bm           is in  tools
    echo ---------------------------------------------------------------
)

endlocal
