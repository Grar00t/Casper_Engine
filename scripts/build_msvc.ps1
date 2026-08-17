<#
.SYNOPSIS
    Build NIYAH-CORE using MSVC (cl.exe).
#>
[CmdletBinding()]
param(
    [ValidateSet('x64','arm64','auto')]
    [string]$Arch = 'auto',
    [ValidateSet('Debug','Release')]
    [string]$Config = 'Release'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$Root = (Get-Location).Path

if ($Arch -eq 'auto') {
    $Arch = if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq [System.Runtime.InteropServices.Architecture]::Arm64) { 'arm64' } else { 'x64' }
}
Write-Host "[build_msvc] Target arch : $Arch"
Write-Host "[build_msvc] Config      : $Config"

function Find-VsDevCmd {
    $vswhere = "$env:ProgramFiles(x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { $vswhere = "$env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe" }
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($LASTEXITCODE -eq 0 -and $vsPath) {
            $candidate = Join-Path $vsPath 'Common7\Tools\VsDevCmd.bat'
            if (Test-Path $candidate) { return $candidate }
        }
    }
    foreach ($fb in @(
        'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat')) {
        if (Test-Path $fb) { return $fb }
    }
    return $null
}

function Test-ArchFlag([string]$flag) {
    $tmp = [System.IO.Path]::GetTempFileName() + '.c'
    try {
        'int main(void){return 0;}' | Set-Content $tmp
        $out = & cmd /c "cl.exe /nologo $flag `"$tmp`" /Fe:nul 2>&1"
        return ($LASTEXITCODE -eq 0 -and ($out -notmatch 'D9002'))
    } finally {
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
}

if ($Arch -eq 'arm64') { $archFlag = '' }
elseif ($Config -eq 'Release' -and (Test-ArchFlag '/arch:AVX2')) { $archFlag = '/arch:AVX2' }
else { $archFlag = '' }

if ($archFlag) { Write-Host "[build_msvc] SIMD flag   : $archFlag" }
else { Write-Host '[build_msvc] SIMD flag   : (none — scalar fallback)' }

$commonFlags = @('/nologo', '/W4', '/WX', '/wd4996', '/EHsc')
if ($archFlag) { $commonFlags += $archFlag }
$configFlags = if ($Config -eq 'Release') { @('/O2', '/GL', '/DNDEBUG') } else { @('/Od', '/Zi', '/RTC1') }

function Invoke-ClBuild {
    param([string[]]$Sources, [string]$Out, [string[]]$ExtraFlags = @(), [string[]]$LinkerLibs = @())
    $allFlags = @($commonFlags + $configFlags + @('/I.', '/ICore_CPP', '/Iinclude') + $ExtraFlags)
    $flagStr = $allFlags -join ' '
    $srcStr = ($Sources | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $libStr = if ($LinkerLibs.Count -gt 0) { ' /link ' + ($LinkerLibs -join ' ') } else { '' }
    $cmd = "cl.exe $flagStr $srcStr /Fe:`"$Out`"$libStr"
    Write-Host "`n[build_msvc] Compiling: $Out"
    Write-Host "  cmd> $cmd"
    $result = & cmd /c "$cmd 2>&1"
    $nestedExit = $LASTEXITCODE
    foreach ($line in $result) { Write-Host "  $line" }
    if ($nestedExit -ne 0) { throw "[build_msvc] FAILED (exit $nestedExit): $Out" }
    if (-not (Test-Path $Out)) { throw "[build_msvc] compiler reported success but artifact is missing: $Out" }
    $sz = (Get-Item $Out).Length
    if ($sz -le 0) { throw "[build_msvc] artifact is empty: $Out" }
    Write-Host "[build_msvc] OK  $Out  ($([math]::Round($sz/1KB,1)) KB)"
}

$clInPath = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $clInPath) {
    $vsDevCmd = Find-VsDevCmd
    if (-not $vsDevCmd) { throw '[build_msvc] cl.exe not found and VsDevCmd.bat not located. Install Visual Studio Build Tools (C++ workload).' }
    Write-Host "[build_msvc] Bootstrapping MSVC environment from:`n  $vsDevCmd"
    $scriptPath = $MyInvocation.MyCommand.Path
    $archArg = "-Arch $Arch"
    $cfgArg = "-Config $Config"
    $inner = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" $archArg $cfgArg"
    & cmd /c "`"$vsDevCmd`" -arch=$Arch -no_logo && $inner"
    if ($LASTEXITCODE -ne 0) { throw "[build_msvc] bootstrapped build failed (exit $LASTEXITCODE)." }
    exit 0
}

$niyahSrc = @("$Root\Core_CPP\niyah_core.c", "$Root\Core_CPP\niyah_main.c")
$niyahTrainSrc = @("$Root\Core_CPP\niyah_train.c", "$Root\Core_CPP\niyah_core.c", "$Root\tokenizer.c")
$hybridSrc = @(
    "$Root\Core_CPP\niyah_hybrid_main.c",
    "$Root\Core_CPP\niyah_core.c",
    "$Root\Core_CPP\hybrid_reasoner.c",
    "$Root\Core_CPP\constraint_solver.c",
    "$Root\Core_CPP\rule_parser.c",
    "$Root\Core_CPP\proof_generator.c",
    "$Root\Core_CPP\khz_q_svd.c",
    "$Root\Core_CPP\casper_rag.c",
    "$Root\tokenizer.c"
)
$benchSrc = @("$Root\Core_CPP\bench_niyah.c", "$Root\Core_CPP\niyah_core.c")

Invoke-ClBuild -Sources $niyahSrc -Out "$Root\Core_CPP\niyah.exe" -ExtraFlags @('/std:c17')
Invoke-ClBuild -Sources $niyahTrainSrc -Out "$Root\niyah_train.exe" -ExtraFlags @('/std:c17')
Invoke-ClBuild -Sources $hybridSrc -Out "$Root\Core_CPP\niyah_hybrid.exe" -ExtraFlags @('/std:c17', '/DWINHTTP_LINK')
Invoke-ClBuild -Sources $benchSrc -Out "$Root\Core_CPP\bench_niyah.exe" -ExtraFlags @('/std:c17')

$casperSrc = @(
    "$Root\Core_CPP\casper_cli.c",
    "$Root\Core_CPP\casper_rag.c",
    "$Root\Core_CPP\rule_parser.c",
    "$Root\Core_CPP\proof_generator.c"
)
Invoke-ClBuild -Sources $casperSrc -Out "$Root\casper.exe" -ExtraFlags @('/std:c17') -LinkerLibs @('winhttp.lib')

Write-Host "`n[build_msvc] Artifact checksums (SHA256):"
foreach ($artifact in @(
    "$Root\Core_CPP\niyah.exe",
    "$Root\niyah_train.exe",
    "$Root\Core_CPP\niyah_hybrid.exe",
    "$Root\Core_CPP\bench_niyah.exe",
    "$Root\casper.exe"
)) {
    if (-not (Test-Path $artifact)) { throw "[build_msvc] required artifact missing: $artifact" }
    $hash = (Get-FileHash $artifact -Algorithm SHA256).Hash
    $sz = [math]::Round((Get-Item $artifact).Length / 1KB, 1)
    Write-Host "  $hash  $((Split-Path $artifact -Leaf))  (${sz} KB)"
}

Write-Host "`n[build_msvc] BUILD PASS (exit 0; all artifacts present)."