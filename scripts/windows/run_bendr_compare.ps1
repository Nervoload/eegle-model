param(
    [string]$CondaEnv = "eegml",
    [string]$Device = "auto",
    [int[]]$Subjects = @(1),
    [string]$BendrRepo = $env:BENDR_REPO,
    [string]$EncoderWeights = $env:BENDR_ENCODER_WEIGHTS,
    [string]$ContextWeights = $env:BENDR_CONTEXT_WEIGHTS,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

conda activate $CondaEnv
pip install -e .

$RunArgs = @(
    "scripts/run_suite.py",
    "--suite", "configs/suites/p300_bendr_compare_windows.yml",
    "--device", $Device,
    "--subjects"
)
foreach ($Subject in $Subjects) {
    $RunArgs += [string]$Subject
}

if ($BendrRepo) {
    $env:BENDR_REPO = (Resolve-Path $BendrRepo).Path
    $RunArgs += @("--bendr-repo", $env:BENDR_REPO)
}
if ($EncoderWeights) {
    $env:BENDR_ENCODER_WEIGHTS = (Resolve-Path $EncoderWeights).Path
    $RunArgs += @("--bendr-encoder", $env:BENDR_ENCODER_WEIGHTS)
}
if ($ContextWeights) {
    $env:BENDR_CONTEXT_WEIGHTS = (Resolve-Path $ContextWeights).Path
    $RunArgs += @("--bendr-context", $env:BENDR_CONTEXT_WEIGHTS)
}
if ($DryRun) {
    $RunArgs += "--dry-run"
}

python @RunArgs
