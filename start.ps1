param([ValidateSet('CPU','GPU')][string]$Mode='CPU')
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$python=$null
if(Get-Command py -ErrorAction SilentlyContinue){
 try {
  $found=& py -3.12 -c "import sys; print(sys.executable)" 2>$null
  if($LASTEXITCODE -eq 0){$python=$found}
 } catch { $python=$null }
}
if(-not $python){
 $candidate=Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
 if(Test-Path $candidate){$python=$candidate}
}
if(-not $python){
 Write-Host 'Python 3.12 is required. Install the official Python package with Windows Package Manager?'
 if((Read-Host '[y/N]') -ne 'y'){exit 1}
 if(-not(Get-Command winget -ErrorAction SilentlyContinue)){
  Write-Host 'Windows Package Manager unavailable. Install Python 3.12 (64-bit) from https://www.python.org/downloads/windows/ then run again.';exit 1
 }
 & winget install --id Python.Python.3.12 --exact --source winget --scope user --silent --accept-package-agreements --accept-source-agreements
 if($LASTEXITCODE -ne 0){throw 'Python installation failed. Check the winget error above.'}
 $python=Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
 if(-not(Test-Path $python)){throw 'Python installed in a different location. Restart this console and rerun, or launch CPU.py/GPU.py with Python 3.12.'}
}
& $python (Join-Path $PSScriptRoot "$Mode.py")
exit $LASTEXITCODE
