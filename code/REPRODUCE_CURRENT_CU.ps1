param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [string]$OutDir = (Join-Path $PSScriptRoot "reproduced"),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$desis = Join-Path $DataDir "desis_2021.csv"
$prisma = Join-Path $DataDir "prisma_2021.csv"
if (-not (Test-Path -LiteralPath $desis -PathType Leaf)) { throw "Missing public input: $desis" }
if (-not (Test-Path -LiteralPath $prisma -PathType Leaf)) { throw "Missing public input: $prisma" }
$expectedDesis = "85ac21e38fb990a75714f90be0d2ced643275adf4e46570f326e765709c48ccc"
$expectedPrisma = "0c6ef3dff0e5eff1c3e9ebb349494465d4459ee189c2aeb089af4e7622822647"
if ((Get-FileHash -LiteralPath $desis -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedDesis) { throw "DESIS input hash mismatch" }
if ((Get-FileHash -LiteralPath $prisma -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedPrisma) { throw "PRISMA input hash mismatch" }

$scripts = Join-Path $PSScriptRoot "scripts"
$cuOut = Join-Path $OutDir "current_common_utility"
$anchorOut = Join-Path $OutDir "current_reference_anchor"
New-Item -ItemType Directory -Force -Path $cuOut, $anchorOut | Out-Null
$threadKeys = @("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
$previous = @{}
foreach ($key in $threadKeys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
    [Environment]::SetEnvironmentVariable($key, "1", "Process")
}
try {
    & $Python (Join-Path $scripts "run_common_utility_expanded.py") --desis $desis --prisma $prisma --out $cuOut --expanded --lock-published
    if ($LASTEXITCODE -ne 0) { throw "Current CU Python process failed with exit code $LASTEXITCODE" }
    & $Python (Join-Path $scripts "run_current_reference_anchor.py") --desis $desis --prisma $prisma --out $anchorOut
    if ($LASTEXITCODE -ne 0) { throw "Reference-anchor Python process failed with exit code $LASTEXITCODE" }
    Write-Host "Current CU reproduction completed: $OutDir"
} finally {
    foreach ($key in $threadKeys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], "Process") }
}
