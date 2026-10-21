# Run fixed-point analysis on a list of checkpoints.
# Usage: .\run_fixed_point_analysis.ps1 [extra args forwarded to fixed_point_analysis.py]
#
# Add or remove entries in $Checkpoints to control which models are analysed.
# Examples:
#   .\run_fixed_point_analysis.ps1
#   .\run_fixed_point_analysis.ps1 --n-iters 12000 --show

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── load TONE_RESULTS_DIR from project .env ───────────────────────────────────
$EnvFile = Join-Path (Split-Path -Parent $ScriptDir) ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.+)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim())
        }
    }
}

$ResultsDir = $env:TONE_RESULTS_DIR
if (-not $ResultsDir) {
    Write-Error "TONE_RESULTS_DIR is not set. Add it to .env or set it in the environment."
    exit 1
}

# ── checkpoints to analyse ────────────────────────────────────────────────────
$Checkpoints = @(
    # pure tone model
    "$ResultsDir\pure\pure_0_lr5e-05_bs128_noise0.5_nrnn128\checkpoint.pkl"

    # white-noise model
    "$ResultsDir\wn\wn_0_lr5e-05_bs128_noise0.5_nrnn128\checkpoint.pkl"

    # harmonic complex tone (HCT) model
    "$ResultsDir\harmonic\harmonic_0_lr5e-05_bs128_noise0.5_nrnn128\checkpoint.pkl"
)

# ── run ───────────────────────────────────────────────────────────────────────
Write-Host "Results dir : $ResultsDir"
Write-Host "Checkpoints : $($Checkpoints.Count)"
Write-Host ""

$Failed = @()

foreach ($Ckpt in $Checkpoints) {
    if (-not (Test-Path $Ckpt)) {
        Write-Host "SKIP (not found): $Ckpt"
        $Failed += $Ckpt
        continue
    }

    Write-Host "================================================================"
    Write-Host "Running: $Ckpt"
    Write-Host "================================================================"

    python "$ScriptDir\fixed_point_analysis.py" $Ckpt @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Analysis failed for: $Ckpt (exit $LASTEXITCODE)"
        $Failed += $Ckpt
    }
}

# ── summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Done. $($Checkpoints.Count) checkpoints attempted."
if ($Failed.Count -gt 0) {
    Write-Host "Failed or skipped:"
    $Failed | ForEach-Object { Write-Host "  $_" }
}
