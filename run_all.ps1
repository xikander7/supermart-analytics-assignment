# Run all pipeline steps on Windows PowerShell
param(
    [switch]$CleanOnly
)
$ErrorActionPreference = "Stop"
# Set PYTHONPATH to repository root so "src" is importable
$env:PYTHONPATH = (Get-Location).Path
Write-Host "PYTHONPATH set to $env:PYTHONPATH"

python scripts\data_cleaning.py
if ($CleanOnly) { exit }

python scripts\feature_engineering.py
python scripts\train_model.py
python scripts\generate_insights.py
