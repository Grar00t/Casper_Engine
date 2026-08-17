param(
    [Parameter(Position = 0)]
    [ValidateSet("build", "corpus", "train", "smoke", "bench", "save", "run", "all")]
    [string]$Action = "all",

    [Parameter(Position = 1)]
    [string]$DataPath = "Data_Training/sovereign_knowledge.txt",

    [Parameter(Position = 2)]
    [int]$Epochs = 3,

    [Parameter(Position = 3)]
    [double]$Lr = 0.001,
    [double]$MinLr = 0.0001,

    [string]$Prompt = "bismillah",
    [int]$Tokens = 64,
    [string]$Model = "niyah_tiny.bin",
    [string]$Size = "tiny",
    [int]$Steps = 200,
    [double]$Temp = 0.8,
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

function Invoke-Build {
    Write-Host "[niyah] build..."
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\build_msvc.ps1") -Config Release
    Assert-ProcessSuccess "build" $LASTEXITCODE
}

function Require-Binaries {
    $hybrid = Join-Path $RepoRoot "Core_CPP\niyah_hybrid.exe"
    if (-not (Test-Path $hybrid)) {
        Invoke-Build
    }
    if (-not (Test-Path $hybrid)) {
        throw "[niyah] required artifact missing: $hybrid"
    }
}

function Invoke-Corpus {
    Write-Host "[niyah] corpus..."
    $corpusScript = Join-Path $RepoRoot "scripts\build_corpus.ps1"
    if (Test-Path $corpusScript) {
        & powershell -ExecutionPolicy Bypass -File $corpusScript
        Assert-ProcessSuccess "corpus" $LASTEXITCODE
    } else {
        Write-Host "[niyah] corpus script not found; skipping."
    }
}

function Invoke-Train {
    Write-Host "[niyah] train..."
    Require-Binaries
    $trainer = Join-Path $RepoRoot "niyah_train.exe"
    if (-not (Test-Path $trainer)) {
        throw "[niyah] required artifact missing: $trainer"
    }
    & $trainer $DataPath $Epochs $Lr $MinLr
    Assert-ProcessSuccess "train" $LASTEXITCODE
}

function Invoke-Smoke {
    Write-Host "[niyah] smoke..."
    Require-Binaries
    $smoke = Join-Path $RepoRoot "Core_CPP\niyah_hybrid.exe"
    if (-not (Test-Path $smoke)) {
        throw "[niyah] smoke executable missing: $smoke"
    }
    & $smoke --smoke
    $exitCode = $LASTEXITCODE
    Assert-ProcessSuccess "smoke" $exitCode
    Write-Host "[niyah] SMOKE PASS (exit 0)"
}

function Invoke-Bench {
    Write-Host "[niyah] bench..."
    Require-Binaries
    $bench = Join-Path $RepoRoot "Core_CPP\bench_niyah.exe"
    if (-not (Test-Path $bench)) {
        Write-Host "[niyah] building benchmark binary..."
        Invoke-Build
    }
    if (-not (Test-Path $bench)) {
        throw "[niyah] benchmark executable missing: $bench"
    }
    & $bench
    Assert-ProcessSuccess "bench" $LASTEXITCODE
}

function Invoke-Save {
    Write-Host "[niyah] save..."
    Invoke-Train
    $saved = Join-Path $RepoRoot "niyah_trained.bin"
    if (-not (Test-Path $saved)) {
        throw "[niyah] model output was not produced: $saved"
    }
    Write-Host "[niyah] saved model to $saved"
}

function Invoke-Run {
    Write-Host "[niyah] run..."
    Require-Binaries
    $hybrid = Join-Path $RepoRoot "Core_CPP\niyah_hybrid.exe"
    if (-not (Test-Path $hybrid)) {
        throw "[niyah] hybrid executable missing: $hybrid"
    }

    $resolvedModel = $Model
    if (-not (Test-Path $resolvedModel)) {
        $fallback = Join-Path $RepoRoot "niyah_trained.bin"
        if (Test-Path $fallback) {
            $resolvedModel = $fallback
            Write-Host "[niyah] model '$Model' not found; using '$resolvedModel'"
        } else {
            throw "[niyah] no model file found: '$Model' or '$fallback'"
        }
    }

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
        if (Test-Path (Join-Path $RepoRoot "Data_Training\sources")) {
            Invoke-Corpus
        }
        Invoke-Train
        Invoke-Smoke
    }
}