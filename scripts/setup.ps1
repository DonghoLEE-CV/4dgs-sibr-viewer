# SIBR-4D one-shot setup (Windows 10/11 x64 + NVIDIA GPU).
# Called by setup.bat. Everything is installed inside this repository folder:
#   env\                      conda environment (Python 3.10, CUDA 12.4 compiler, PyTorch 2.5.1 cu124, cmake, ninja)
#   third_party\sibr_core\    SIBR (Inria) at a pinned commit + the SIBR-4D patch, built into install\bin
#   third_party\4DGaussians\  4DGaussians (hustvl) at a pinned commit (used only by export_4dgs.bat)
# Re-running is safe: finished steps are skipped. Use -Clean to rebuild the viewer from scratch.
param([switch]$Clean, [switch]$Yes)

$ErrorActionPreference = "Continue"   # native tools write to stderr; exit codes are checked explicitly
$Root = Split-Path -Parent $PSScriptRoot
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Env = Join-Path $Root "env"
$TP = Join-Path $Root "third_party"
$Sibr = Join-Path $TP "sibr_core"
$G4 = Join-Path $TP "4DGaussians"

$SIBR_URL = "https://gitlab.inria.fr/sibr/sibr_core.git"
$SIBR_BRANCH = "gaussian_code_release_union"
$SIBR_COMMIT = "4c4a95365597a78b105792794db70f09a1ece938"
$G4_URL = "https://github.com/hustvl/4DGaussians.git"
$G4_COMMIT = "843d5ac636c37e4b611242287754f3d4ed150144"
$CUDA_CHANNEL = "nvidia/label/cuda-12.4.1"

function Step($t) { Write-Host ""; Write-Host "==== $t" -ForegroundColor Cyan }
function Ok($t) { Write-Host "  [ok] $t" -ForegroundColor Green }
function Warn($t) { Write-Host "  [!] $t" -ForegroundColor Yellow }
function Fail($t) { Write-Host ""; Write-Host "  [FAILED] $t" -ForegroundColor Red; Write-Host "  logs: $Logs"; exit 1 }
function Ask($q) { if ($Yes) { return $true }; $a = Read-Host "  $q [y/N]"; return ($a -match '^(y|yes)$') }
function Run($exe, [string[]]$argv, $log) {
    "`n>>> $exe $($argv -join ' ')" | Out-File -Append -Encoding utf8 (Join-Path $Logs $log)
    & $exe @argv 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $Logs $log)
    if ($LASTEXITCODE -ne 0) { Fail "$exe $($argv -join ' ')   (see logs\$log)" }
}
function HasWinget { return [bool](Get-Command winget -ErrorAction SilentlyContinue) }

# ------------------------------------------------------------------ 1. prerequisites
Step "1/6  Checking prerequisites"
if (-not [Environment]::Is64BitOperatingSystem) { Fail "64-bit Windows is required" }

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { Fail "NVIDIA driver not found (nvidia-smi). Install the latest driver from https://www.nvidia.com/drivers" }
$gpu = (& nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | Select-Object -First 1)
$cc = ($gpu.Split(',')[1]).Trim()
$arch = $cc.Replace('.', '')
if ([int]$arch -lt 70) { Fail "GPU compute capability $cc < 7.0 is not supported by the 3DGS rasterizer" }
Ok "GPU: $gpu  (CUDA arch $arch)"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Warn "Git is not installed."
    if ((HasWinget) -and (Ask "Install Git with winget now?")) { winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements; Fail "Git installed - close this window and run setup.bat again" }
    Fail "Install Git from https://git-scm.com/download/win and run setup.bat again"
}
Ok "git: $((git --version) -join ' ')"

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vs = $null
if (Test-Path $vswhere) { $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath }
if (-not $vs) {
    Warn "Visual Studio C++ build tools (MSVC x64) not found."
    if ((HasWinget) -and (Ask "Install 'Visual Studio 2022 Build Tools' (C++ workload, ~6 GB) with winget now?")) {
        winget install -e --id Microsoft.VisualStudio.2022.BuildTools --accept-package-agreements --accept-source-agreements --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        Fail "Build tools installed - run setup.bat again"
    }
    Fail "Install Visual Studio 2022 (Community or Build Tools) with the 'Desktop development with C++' workload, then run setup.bat again"
}
# CUDA 12.4 accepts MSVC up to 14.4x (VS 2022). Prefer such a toolset if several are installed.
$tools = Get-ChildItem (Join-Path $vs "VC\Tools\MSVC") -Directory | Sort-Object Name -Descending
$tool = $tools | Where-Object { $_.Name -match '^14\.(3|4)\d\.' } | Select-Object -First 1
$unsupported = $false
if (-not $tool) { $tool = $tools | Select-Object -First 1; $unsupported = $true; Warn "MSVC $($tool.Name) is newer than CUDA 12.4 supports; building with -allow-unsupported-compiler" }
$vcvarsVer = ($tool.Name -split '\.')[0..1] -join '.'
Ok "Visual Studio: $vs  (MSVC $($tool.Name))"

$conda = $null
$c = Get-Command conda -ErrorAction SilentlyContinue
if ($c) { $conda = $c.Source }
if (-not $conda) {
    foreach ($p in @("$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3", "$env:LOCALAPPDATA\miniconda3", "$env:ProgramData\miniconda3", "$env:ProgramData\anaconda3", "C:\miniconda3", "C:\anaconda3", "D:\anaconda3", "D:\miniconda3")) {
        if (Test-Path "$p\Scripts\conda.exe") { $conda = "$p\Scripts\conda.exe"; break }
    }
}
if (-not $conda) {
    Warn "conda (Miniconda / Anaconda) not found."
    if ((HasWinget) -and (Ask "Install Miniconda with winget now?")) { winget install -e --id Anaconda.Miniconda3 --accept-package-agreements --accept-source-agreements; Fail "Miniconda installed - run setup.bat again" }
    Fail "Install Miniconda from https://docs.anaconda.com/miniconda/ and run setup.bat again"
}
if ($conda -notlike "*.exe") {   # e.g. anaconda3\Library\bin\conda.bat or condabin\conda.bat -> Scripts\conda.exe
    $d = Split-Path $conda
    for ($i = 0; $i -lt 3 -and $d; $i++) { if (Test-Path "$d\Scripts\conda.exe") { $conda = "$d\Scripts\conda.exe"; break }; $d = Split-Path $d }
}
Ok "conda: $conda"

$free = (Get-PSDrive -Name (Split-Path -Qualifier $Root).TrimEnd(':')).Free / 1GB
if ($free -lt 15) { Warn ("only {0:N0} GB free on this drive; about 12 GB is needed" -f $free) }

# ------------------------------------------------------------------ 2. python / CUDA environment
Step "2/6  Python + CUDA environment (env\)"
$py = Join-Path $Env "python.exe"
if (-not (Test-Path $py)) {
    Run $conda @("create", "-y", "-p", $Env, "--override-channels", "-c", "conda-forge", "python=3.10") "conda.log"
}
Ok "python: $((& $py --version) -join ' ')"
if (-not (Test-Path (Join-Path $Env "bin\nvcc.exe"))) {
    Write-Host "  installing CUDA 12.4 toolkit (conda, ~3 GB download) ..."
    Run $conda @("install", "-y", "-p", $Env, "--override-channels", "-c", $CUDA_CHANNEL, "-c", "conda-forge", "cuda-toolkit=12.4.1") "conda.log"
}
Ok "nvcc: $((& (Join-Path $Env 'bin\nvcc.exe') --version | Select-String 'release') -join ' ')"
Write-Host "  installing Python packages from requirements.txt (PyTorch 2.5.1 cu124 is ~2.5 GB the first time) ..."
Run $py @("-m", "pip", "install", "-r", (Join-Path $Root "requirements.txt")) "pip.log"
& $py -c "import torch; assert torch.cuda.is_available(); print('  [ok] torch', torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { Fail "PyTorch cannot use the GPU (update the NVIDIA driver)" }

# ------------------------------------------------------------------ 3. sources
Step "3/6  Sources (pinned commits)"
New-Item -ItemType Directory -Force -Path $TP | Out-Null
if (-not (Test-Path (Join-Path $Sibr ".git"))) { Run git @("clone", "--branch", $SIBR_BRANCH, $SIBR_URL, $Sibr) "git.log" }
Run git @("-C", $Sibr, "checkout", "-q", "--force", $SIBR_COMMIT) "git.log"
Ok "SIBR $SIBR_COMMIT"
if (-not (Test-Path (Join-Path $G4 ".git"))) { Run git @("clone", $G4_URL, $G4) "git.log" }
Run git @("-C", $G4, "checkout", "-q", "--force", $G4_COMMIT) "git.log"
Ok "4DGaussians $G4_COMMIT"

# ------------------------------------------------------------------ 4. SIBR-4D patch
Step "4/6  Applying the SIBR-4D patch"
$patch = Join-Path $Root "sibr_patch"
Get-ChildItem $patch -Recurse -File | Where-Object { $_.Extension -ne ".diff" } | ForEach-Object {
    $rel = $_.FullName.Substring($patch.Length + 1)
    $dst = Join-Path $Sibr $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
    Copy-Item $_.FullName $dst -Force
}
Ok "patched files copied into third_party\sibr_core"

# ------------------------------------------------------------------ 5. build
Step "5/6  Building the viewer (first time: dependency download ~0.5 GB + 10-20 min compile)"
$bld = Join-Path $Sibr "build"
if ($Clean -and (Test-Path $bld)) { Remove-Item -Recurse -Force $bld }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$shim = Join-Path $Root "scripts\7z_shim.bat"
$cudaFlags = "-Xcompiler /utf-8"
if ($unsupported) { $cudaFlags += " -allow-unsupported-compiler" }
$cmd = @"
@echo off
call "$vcvars" -vcvars_ver=$vcvarsVer > nul || exit /b 1
set "PATH=$Env\Scripts;$Env\bin;$Env\Library\bin;%PATH%"
set "CUDA_PATH=$Env"
cmake -S "$Sibr" -B "$bld" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DCMAKE_CUDA_COMPILER="$Env/bin/nvcc.exe" -DCUDAToolkit_ROOT="$Env" -DCMAKE_CUDA_ARCHITECTURES=$arch ^
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DSEVEN_ZIP_CMD="$shim" ^
  -DCMAKE_CUDA_FLAGS="$cudaFlags" -DCMAKE_CXX_FLAGS="/utf-8 /EHsc /DNOMINMAX" ^
  -DBUILD_IBR_GAUSSIANVIEWER=ON -DBUILD_IBR_REMOTE=OFF -DBUILD_IBR_ULR=OFF -DBUILD_IBR_DATASET_TOOLS=OFF -DBUILD_IBR_BASIC=ON || exit /b 1
cmake --build "$bld" --target install -j %NUMBER_OF_PROCESSORS% || exit /b 1
"@
$bat = Join-Path $Logs "build_cmd.bat"
Set-Content -Path $bat -Value $cmd -Encoding ascii
cmd /c "`"$bat`" > `"$Logs\build.log`" 2>&1"
if ($LASTEXITCODE -ne 0) { Fail "build failed - see logs\build.log (search for 'error')" }
$bin = Join-Path $Sibr "install\bin"
foreach ($d in @("mrf", "xatlas")) { Get-ChildItem (Join-Path $Sibr "extlibs\$d\build") -Filter "*.dll" -ErrorAction SilentlyContinue | Copy-Item -Destination $bin -Force }
$exe = Join-Path $bin "SIBR_gaussian4D_app_rwdi.exe"
if (-not (Test-Path $exe)) { Fail "viewer executable not produced - see logs\build.log" }
Ok "built $exe"

# ------------------------------------------------------------------ 6. smoke test
Step "6/6  Smoke test"
$env:PATH = "$bin;$Env\bin;$Env\Library\bin;" + $env:PATH
$pr = Start-Process -FilePath $exe -WorkingDirectory $bin -PassThru -WindowStyle Minimized
Start-Sleep -Seconds 8
if ($pr.HasExited) {
    if ($pr.ExitCode -eq -1073741515) { Fail "a DLL is missing when starting the viewer (0xC0000135)" }
    Fail "the viewer exited right away (code $($pr.ExitCode))"
}
Stop-Process -Id $pr.Id -Force
Ok "viewer window opens"
Write-Host ""
Write-Host "SIBR-4D is ready." -ForegroundColor Green
Write-Host "  view a model      : run_viewer.bat <model folder> [<model folder> ...]   (or drag folders onto the window)"
Write-Host "  convert 4DGaussians: export_4dgs.bat --ckpt <output folder> --data <scene folder> --out <model folder>"
