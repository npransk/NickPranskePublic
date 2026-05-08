param(
    [int]$MinNewEvents = 100,
    [int]$Episodes = 2000,
    [int]$EvalGames = 100
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "nightly_pipeline_$Stamp.log"

if (Test-Path ".\.venv\Scripts\python.exe") {
    $Python = ".\.venv\Scripts\python.exe"
} else {
    $Python = "python"
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"

& $Python -m yahtzee_mlops.cli pipeline `
    --min-new-events $MinNewEvents `
    --episodes-per-run $Episodes `
    --eval-games $EvalGames `
    *> $LogPath

Get-Content $LogPath
