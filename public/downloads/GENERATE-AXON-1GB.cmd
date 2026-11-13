@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "AXON-Corpus-1GB.py"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    python "AXON-Corpus-1GB.py"
  ) else (
    echo Python 3 is required to create the local training corpus.
    echo No files were installed.
    exit /b 1
  )
)
if errorlevel 1 exit /b 1
echo AXON training corpus generated.
pause
