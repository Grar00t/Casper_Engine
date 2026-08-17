param(
    [Parameter(Position = 0)]
    [ValidateSet("build", "corpus", "train", "smoke", "bench", "save", "run", "all")]
    [string]$Action = "all",

    [Parameter(Position = 1)]
    [string]$DataPath = "Data_Training/sovereign_knowledge.txt",

    [Parameter(Position = 2)]
    [ValidateRange(1, 1000000)]
    [int]$Epochs = 3,

    [Parameter(Position = 3)]
    [ValidateRange(0.000000001, 100.0)]
    [double]$Lr = 0.001,
    [ValidateRange(0.000000001, 100.0)]
    [double]$MinLr = 0.0001,

    [ValidateLength(1, 65536)]
    [string]$Prompt = "bismillah",
    [ValidateRange(1, 1048576)]
    [int]$Tokens = 64,
    [string]$Model = "niyah_tiny.bin",
    [string]$Size = "tiny",
    [ValidateRange(1, 100000000)]
    [int]$Steps = 200,
    [ValidateRange(0.000001, 100.0)]
    [double]$Temp = 0.8,
    [ValidateRange(0.000001, 1.0)]
    [double]$TopP = 0.9,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$RepoRoot = (Get-Location).Path

function Assert-ProcessSuccess([string]$Name, [int]$ExitCode) {
    if ($ExitCode -ne 0) {
        throw "[niyah] $Name failed (exit $ExitCode)."
    }
}

function Assert-Artifact([string]$Path, [string]$Name) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "[niyah] $Name artifact missing: $Path"
    }
    $file = Get-Item -LiteralPath $Path
    if ($file.Length -le 0) {
        throw "[niyah] $Name artifact is empty: $Path"
    }
}

function Invoke-Build {
    Write-Host "[niyah] build..."
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\build_msvc.ps1") -Config Release
    Assert-ProcessSuccess "build" $LASTEXITCODE
    Assert-Artifact (Join-Path $RepoRoot "Core_CPP\niyah.exe") "niyah"
    Assert-Artifact (Join-Path $RepoRoot "niyah_train.exe") "trainer"
    Assert-Artifact (Join-Path $RepoRoot "Core_CPP\niyah_hybrid.exe") "hybrid"
    Assert-Artifact (Join-Path $RepoRoot "Core_CPP\bench_niyah.exe") "benchmark"
}

function Require-Binaries {
    $hybrid = Join-Path $RepoRoot "Core_CPP\niyah_hybrid.exe"
    if (-not (Test-Path -LiteralPath $hybrid -PathType Leaf)) {
        Invoke-Build
    }
    Assert-Artifact $hybrid "hybrid"
}

function Invoke-Corpus {
    Write-Host "[niyah] corpus..."
    $corpusScript = Join-Path $RepoRoot "scripts\build_corpus.ps1"
    if (-not (Test-Path -LiteralPath $corpusScript -PathType Leaf)) {
        throw "[niyah] corpus script missing: $corpusScript"
    }
    & powershell -ExecutionPolicy Bypass -File $corpusScript
    Assert-ProcessSuccess "corpus" $LASTEXITCODE
}

function Invoke-Train {
    Write-Host "[niyah] train..."
    Require-Binaries
    $trainer = Join-Path $RepoRoot "niyah_train.exe"
    Assert-Artifact $trainer "trainer"
    if (-not (Test-Path -LiteralPath $DataPath -PathType Leaf)) {
        throw "[niyah] training data missing: $DataPath"
    }
    & $trainer $DataPath $Epochs $Lr $MinLr
    Assert-ProcessSuccess "train" $LASTEXITCODE
}

function Invoke-Smoke {
    Write-Host "[niyah] smoke..."
    Require-Binaries
    $smoke = Join-Path $RepoRoot "Core_CPP\niyah_hybrid.exe"
    Assert-Artifact $smoke "smoke"
    & $smoke --smoke
    $exitCode = $LASTEXITCODE
    Assert-ProcessSuccess "smoke" $exitCode
    Write-Host "[niyah] SMOKE PASS (exit 0)"
}

function Invoke-Bench {
    Write-Host "[niyah] bench..."
    Require-Binaries
    $bench = Join-Path $RepoRoot "Core_CPP\bench_niyah.exe"
    if (-not (Test-Path -LiteralPath $bench -PathType Leaf)) {
        Invoke-Build
    }
    Assert-Artifact $bench "benchmark"
    & $bench
    Assert-ProcessSuccess "bench" $LASTEXITCODE
}

function Invoke-Save {
    Write-Host "[niyah] save..."
    Invoke-Train
    $saved = Join-Path $RepoRoot "niyah_trained.bin"
    Assert-Artifact $saved "trained model"
    Write-Host "[niyah] saved model to $saved"
}

function Invoke-Run {
    Write-Host "[niyah] run..."
    Require-Binaries
    $hybrid = Join-Path $RepoRoot "Core_CPP\niyah_hybrid.exe"

    $resolvedModel = $Model
    if (-not (Test-Path -LiteralPath $resolvedModel -PathType Leaf)) {
        $fallback = Join-Path $RepoRoot "niyah_trained.bin"
        if (Test-Path -LiteralPath $fallback -PathType Leaf) {
            $resolvedModel = $fallback
        } else {
            throw "[niyah] model missing: '$Model' and '$fallback'"
        }
    }
    Assert-Artifact $resolvedModel "model"

    $inputText = @($Prompt, "quit") -join [Environment]::NewLine
    $inputText | & $hybrid --model $resolvedModel --interactive
    Assert-ProcessSuccess "run" $LASTEXITCODE
}

switch ($Action) {
    "build" { Invoke-Build }
    "corpus" { Invoke-Corpus }
    "train" { Invoke-Train }
    "smoke" { Invoke-Smoke }
    "bench" { Invoke-Bench }
    "save" { Invoke-Save }
    "run" { Invoke-Run }
    "all" {
        Invoke-Build
        if (Test-Path -LiteralPath (Join-Path $RepoRoot "Data_Training\sources") -PathType Container) {
            Invoke-Corpus
        }
        Invoke-Train
        Invoke-Smoke
    }
}
