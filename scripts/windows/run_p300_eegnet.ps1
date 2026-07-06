param(
    [string]$CondaEnv = "eegml",
    [string]$Device = "auto",
    [int[]]$Subjects = @(1),
    [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

conda activate $CondaEnv
pip install -e .

python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

if ($SmokeOnly) {
    python scripts/train.py --config configs/eegnet_p300_bnci2014_008.yml --smoke --device $Device
} else {
    $SubjectArgs = @()
    foreach ($Subject in $Subjects) {
        $SubjectArgs += [string]$Subject
    }
    python scripts/run_suite.py --suite configs/suites/p300_eegnet_windows.yml --subjects $SubjectArgs --device $Device
}
