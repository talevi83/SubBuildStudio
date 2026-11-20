<#
.SYNOPSIS
  Collects everything the Windows installer ships into one folder (build\windows\app by default).

.DESCRIPTION
  1. The source tree of -Ref, exactly as a GitHub release zip has it (git archive, so VERSION is
     filled in from the release tag by export-subst).
  2. Without the development-only parts (tests, QA suite, CI files, run/install scripts, the
     frontend sources) - but with the interface already built (frontend\dist, run
     "npm run build" in frontend\ first), so users need no Node.js.
  3. Plus ffmpeg/ffprobe (packaging\ffmpeg), uv (tools\uv.exe), Python 3.12 (python\), and the
     INSTALLER marker that tells the updater this copy is updated by downloading a new installer.

  Python is python.org's own build (signed), shipped as plain files: letting uv download Python on
  the user's machine creates a directory junction that Windows 11's RedirectionGuard can refuse
  ("untrusted mount point", os error 448), and its unsigned pythonw.exe is blocked by
  behaviour-based antivirus.

  Prints the version (e.g. 2.2.0) and, on GitHub Actions, sets the step output "version".

.EXAMPLE
  cd frontend; npm ci; npm run build; cd ..
  .\packaging\windows\stage.ps1
  & "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" /DAppVersion=2.2.0 packaging\windows\SubBuildStudio.iss
#>
param(
    [string]$OutDir = "build\windows",
    [string]$Ref = "HEAD",
    # Tried in order (each up to 3 times): gyan.dev is a small site that sometimes answers 503
    [string[]]$FfmpegZipUrls = @(
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    ),
    [string]$UvZipUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip",
    # 3.12.10 is the last 3.12 release with Windows binaries from python.org
    [string]$PythonVersion = "3.12.10"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar

# Downloads $Urls[0] to $OutFile, retrying, then falls back to the next URL
function Get-Download([string[]]$Urls, [string]$OutFile) {
    foreach ($url in $Urls) {
        for ($try = 1; $try -le 3; $try++) {
            try {
                Invoke-WebRequest $url -OutFile $OutFile
                return
            } catch {
                Write-Warning "Download failed ($try/3) from ${url}: $($_.Exception.Message)"
                Start-Sleep -Seconds (10 * $try)
            }
        }
    }
    throw "Could not download $OutFile from: $($Urls -join ', ')"
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not [System.IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $repo $OutDir }
$app = Join-Path $OutDir "app"
$downloads = Join-Path $OutDir "downloads"

$dist = Join-Path $repo "frontend\dist\index.html"
if (-not (Test-Path $dist)) { throw "frontend\dist is missing - run 'npm ci' and 'npm run build' in frontend\ first." }

if (Test-Path $app) { Remove-Item $app -Recurse -Force }
New-Item -ItemType Directory -Force $app, $downloads | Out-Null

# 1. The source tree as a release zip has it
Write-Host "Exporting $Ref..."
# (zip, not tar: on the GitHub runner "tar" may be Git's GNU tar, which mistakes C:\ for a host)
$srcZip = Join-Path $downloads "source.zip"
git -C $repo archive --format=zip -o $srcZip $Ref
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
Expand-Archive $srcZip $app -Force

# 2. Development-only parts are not shipped
$devOnly = @(
    "qa-automation-java", "backend\tests", "backend\pytest.ini", ".github", "docs", "Jenkinsfile",
    "SPEC.md", "README.md", "install.bat", "install.sh", "install.command", "run.bat", "run.sh",
    "run.command", "release-tag.bat", "frontend"
)
foreach ($rel in $devOnly) {
    $path = Join-Path $app $rel
    if (Test-Path $path) { Remove-Item $path -Recurse -Force }
}
New-Item -ItemType Directory (Join-Path $app "frontend") | Out-Null
Copy-Item (Join-Path $repo "frontend\dist") (Join-Path $app "frontend\dist") -Recurse

# 3a. ffmpeg + ffprobe
Write-Host "Downloading ffmpeg..."
$ffZip = Join-Path $downloads "ffmpeg.zip"
Get-Download $FfmpegZipUrls $ffZip
$ffDir = Join-Path $downloads "ffmpeg"
Expand-Archive $ffZip $ffDir -Force
$ffExe = Get-ChildItem $ffDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
if (-not $ffExe) { throw "ffmpeg.exe not found in the downloaded ffmpeg archive" }
$ffTarget = Join-Path $app "packaging\ffmpeg"
New-Item -ItemType Directory -Force $ffTarget | Out-Null
Copy-Item (Join-Path $ffExe.DirectoryName "ffmpeg.exe"), (Join-Path $ffExe.DirectoryName "ffprobe.exe") $ffTarget
$ffLicense = Get-ChildItem $ffDir -Recurse -Filter "LICENSE*" | Select-Object -First 1
if ($ffLicense) { Copy-Item $ffLicense.FullName (Join-Path $ffTarget "LICENSE.txt") }

# 3b. uv
Write-Host "Downloading uv..."
$uvZip = Join-Path $downloads "uv.zip"
Get-Download @($UvZipUrl) $uvZip
$uvDir = Join-Path $downloads "uv"
Expand-Archive $uvZip $uvDir -Force
$uvExe = Get-ChildItem $uvDir -Recurse -Filter "uv.exe" | Select-Object -First 1
if (-not $uvExe) { throw "uv.exe not found in $UvZipUrl" }
New-Item -ItemType Directory -Force (Join-Path $app "tools") | Out-Null
Copy-Item $uvExe.FullName (Join-Path $app "tools\uv.exe")

# 3c. Python from python.org (the official NuGet package): python.exe and pythonw.exe are signed by
#     the Python Software Foundation. Behaviour-based antivirus (Bitdefender Advanced Threat Defense)
#     blocked the unsigned python-build-standalone pythonw.exe that uv downloads ("Unsigned process
#     pythonw.exe executed unsigned file pythonw.exe"). The app runs on this Python directly, with no
#     venv: the NuGet package has no venv launchers, and a launcher in front of pythonw.exe would be
#     one more process for the antivirus to judge.
Write-Host "Downloading Python $PythonVersion (python.org NuGet package)..."
$nupkg = Join-Path $downloads "python.zip"
Get-Download @("https://www.nuget.org/api/v2/package/python/$PythonVersion") $nupkg
$pyExtract = Join-Path $downloads "python-nuget"
Expand-Archive $nupkg $pyExtract -Force
$pyTarget = Join-Path $app "python"
Copy-Item (Join-Path $pyExtract "tools") $pyTarget -Recurse
foreach ($rel in @("python.exe", "pythonw.exe")) {
    if (-not (Test-Path (Join-Path $pyTarget $rel))) { throw "python\$rel is missing from the Python $PythonVersion NuGet package" }
}
foreach ($exe in @((Join-Path $pyTarget "python.exe"), (Join-Path $pyTarget "pythonw.exe"))) {
    $sig = Get-AuthenticodeSignature $exe
    if ($sig.Status -ne "Valid") { throw "$exe is not validly signed ($($sig.Status))" }
    Write-Host ("  signed: {0} ({1})" -f (Split-Path $exe -Leaf), $sig.SignerCertificate.Subject)
}

# 3d. The updater checks for this file (backend/app/core/updater.py, INSTALLER_MARKER)
Set-Content -Path (Join-Path $app "INSTALLER") -Value "windows" -Encoding ascii

Remove-Item $downloads -Recurse -Force

# The version from VERSION ("v2.2.0" or "v2.2.0-3-gabc1234"); 0.0.0 for an untagged build
$raw = (Get-Content (Join-Path $app "VERSION") -Raw).Trim()
$version = if ($raw -match '^v?(\d+\.\d+\.\d+)') { $Matches[1] } else { "0.0.0" }
if ($env:GITHUB_OUTPUT) { "version=$version" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8 }
Write-Host "Staged SubBuild Studio $version in $app"
Write-Output $version
