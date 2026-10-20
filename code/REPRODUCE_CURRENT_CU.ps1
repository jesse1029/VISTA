param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [string]$OutDir = (Join-Path $PSScriptRoot "reproduced"),
    [string]$Python = "python"
)
# Reproduce the principal comparison of the manuscript (sequential; see reproduce.sh for a parallel version).
$ErrorActionPreference = "Stop"
$desis = Join-Path $DataDir "desis_2021.csv"
$prisma = Join-Path $DataDir "prisma_2021.csv"
if ((Get-FileHash -LiteralPath $desis -Algorithm SHA256).Hash.ToLowerInvariant() -ne "85ac21e38fb990a75714f90be0d2ced643275adf4e46570f326e765709c48ccc") { throw "DESIS input hash mismatch" }
if ((Get-FileHash -LiteralPath $prisma -Algorithm SHA256).Hash.ToLowerInvariant() -ne "0c6ef3dff0e5eff1c3e9ebb349494465d4459ee189c2aeb089af4e7622822647") { throw "PRISMA input hash mismatch" }
foreach ($key in @("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")) { [Environment]::SetEnvironmentVariable($key, "1", "Process") }
$orig = Join-Path $PSScriptRoot "vendor\original_common_utility.py"
$s = Join-Path $PSScriptRoot "scripts"
$base = Join-Path $OutDir "base"; $ext = Join-Path $OutDir "ext"
if ((Test-Path -LiteralPath $OutDir) -and (Get-ChildItem -LiteralPath $OutDir -Force | Select-Object -First 1)) { throw "Output directory $OutDir is not empty; choose a new one so old results cannot mix in." }
New-Item -ItemType Directory -Force -Path $base, $ext | Out-Null
function Run($script, [string[]]$rest) { & $Python (Join-Path $s $script) @rest; if ($LASTEXITCODE -ne 0) { throw "$script failed" } }
Run "run_vista_comparison.py" @("--original", $orig, "--desis", $desis, "--prisma", $prisma, "--out", $base)
Run "run_additional_harmonizers.py" @("--original", $orig, "--desis", $desis, "--prisma", $prisma, "--out", $ext)
Run "crossed_scale_analysis.py" @($orig, $desis, $prisma, $base, (Join-Path $base "crossed_scale.json"))
Run "controlled_ablation.py" @($orig, $desis, $prisma, $base, (Join-Path $base "controlled_ablation.json"))
Run "reference_anchor.py" @($orig, $desis, $prisma, $base, (Join-Path $OutDir "reference_anchor.json"))
Run "paper_numbers.py" @($OutDir, $desis, (Join-Path $OutDir "paper_numbers.json"))
Run "supp_extra_numbers.py" @($OutDir, (Join-Path $base "split_manifest.csv"), $desis, (Join-Path $OutDir "supp_extra.json"))
Write-Host "Done: $OutDir"
