param(
    [Parameter(Position = 0)]
    [ValidateSet("build", "corpus", "train", "smoke", "bench", "save", "run", "all")]
    [string]$Action = "all",
    [Parameter(Position = 1)] [string]$DataPath = "Data_Training/sovereign_knowledge.txt",
    [Parameter(Position = 2)] [int]$Epochs = 3,
    [Parameter(Position = 3)] [double]$Lr = 0.001,
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
    if ($ExitCode -ne 0) { throw "[niyah] $Name failed (exit $ExitCode)." }
}

function Invoke-Build {
    $script = Join-Path $RepoRoot "scripts\build_gcc.sh"
    if (-not (Test-Path $script)) { throw "[niyah] GCC build script missing: $script" }
    if (-not (Get-Command bash -ErrorAction SilentlyContinue)) { throw "[niyah] bash is required for the native build path." }
    & bash $script --release
    Assert-ProcessSuccess "build" $LASTEXITCODE
}

function Require-Binaries {
    $hybrid = Join-Path $RepoRoot "Core_CPP\niyah_hybrid.exe"
    if (-not (Test-Path $hybrid)) { Invoke-Build }
    if (-not (Test-Path $hybrid)) { throw "[niyah] required artifact missing: $hybrid" }
}

function Invoke-Corpus {
    $corpusScript = Join-Path $RepoRoot "scripts\build_corpus.ps1"
    if (Test-Path $corpusScript) {
        & powershell -ExecutionPolicy Bypass -File $corpusScript
        Assert-ProcessSuccess "corpus" $LASTEXITCODE
    }
}

function Invoke-Train {
    Require-Binaries
    $trainer = Join-Path $RepoRoot "Core_CPP\trainer"
    if (-not (Test-Path $trainer)) { Invoke-Build }
    if (-not (Test-Path $trainer)) { throw "[niyah] trainer artifact missing: $trainer" }
    & $trainer
    Assert-ProcessSuccess "train" $LASTEXITCODE
}

function Invoke-Smoke {
    Require-Binaries
    $smoke = Join-Path $RepoRoot "Core_CPP\niyah"
    if (-not (Test-Path $smoke)) { throw "[niyah] smoke executable missing: $smoke" }
    & $smoke
    Assert-ProcessSuccess "smoke" $LASTEXITCODE
}

function Invoke-Bench {
    Require-Binaries
    $bench = Join-Path $RepoRoot "Core_CPP\bench_niyah.exe"
    if (-not (Test-Path $bench)) { throw "[niyah] benchmark executable missing: $bench" }
    & $bench
    Assert-ProcessSuccess "bench" $LASTEXITCODE
}

function Invoke-Save { Invoke-Train }

function Invoke-Run {
    Require-Binaries
    $hybrid = Join-Path $RepoRoot "Core_CPP\niyah_hybrid.exe"
    if (-not (Test-Path $hybrid)) { throw "[niyah] hybrid executable missing: $hybrid" }
    if (-not (Test-Path $Model)) { throw "[niyah] model missing: $Model" }
    $inputText = @($Prompt, "quit") -join [Environment]::NewLine
    $inputText | & $hybrid --model $Model --interactive
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
    "all" { Invoke-Build; Invoke-Train; Invoke-Smoke }
}
