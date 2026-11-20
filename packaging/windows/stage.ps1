<#
.SYNOPSIS
  Collects everything the Windows installer ships into one folder (build\windows\app by default).

.DESCRIPTION
  1. The source tree of -Ref, exactly as a GitHub release zip has it (git archive, so VERSION is
     filled in from the release tag by export-subst).
  2. Without the development-only parts (tests, QA suite, CI files, run/install scripts, the
     frontend sources) - but with the interface already built (frontend\dist, run
     "npm run build" in frontend\ first), so users need no Node.js.
  3. Plus ffmpeg/ffprobe (packaging\ffmpeg), uv (tools\uv.exe), and the INSTALLER marker that
     tells the updater this copy is updated by downloading a new installer.

  Prints the version (e.g. 2.2.0) and, on GitHub Actions, sets the step output "version".

.EXAMPLE
  cd frontend; npm ci; npm run build; cd ..
  .\packaging\windows\stage.ps1
  & "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" /DAppVersion=2.2.0 packaging\windows\SubBuildStudio.iss
#>
param(
    [string]$OutDir = "build\windows",
    [string]$Ref = "HEAD",
    [string]$FfmpegZipUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    [string]$UvZipUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar

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
Invoke-WebRequest $FfmpegZipUrl -OutFile $ffZip
$ffDir = Join-Path $downloads "ffmpeg"
Expand-Archive $ffZip $ffDir -Force
$ffExe = Get-ChildItem $ffDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
if (-not $ffExe) { throw "ffmpeg.exe not found in $FfmpegZipUrl" }
$ffTarget = Join-Path $app "packaging\ffmpeg"
New-Item -ItemType Directory -Force $ffTarget | Out-Null
Copy-Item (Join-Path $ffExe.DirectoryName "ffmpeg.exe"), (Join-Path $ffExe.DirectoryName "ffprobe.exe") $ffTarget
$ffLicense = Get-ChildItem $ffDir -Recurse -Filter "LICENSE*" | Select-Object -First 1
if ($ffLicense) { Copy-Item $ffLicense.FullName (Join-Path $ffTarget "LICENSE.txt") }

# 3b. uv
Write-Host "Downloading uv..."
$uvZip = Join-Path $downloads "uv.zip"
Invoke-WebRequest $UvZipUrl -OutFile $uvZip
$uvDir = Join-Path $downloads "uv"
Expand-Archive $uvZip $uvDir -Force
$uvExe = Get-ChildItem $uvDir -Recurse -Filter "uv.exe" | Select-Object -First 1
if (-not $uvExe) { throw "uv.exe not found in $UvZipUrl" }
New-Item -ItemType Directory -Force (Join-Path $app "tools") | Out-Null
Copy-Item $uvExe.FullName (Join-Path $app "tools\uv.exe")

# 3c. The updater checks for this file (backend/app/core/updater.py, INSTALLER_MARKER)
Set-Content -Path (Join-Path $app "INSTALLER") -Value "windows" -Encoding ascii

Remove-Item $downloads -Recurse -Force

# The version from VERSION ("v2.2.0" or "v2.2.0-3-gabc1234"); 0.0.0 for an untagged build
$raw = (Get-Content (Join-Path $app "VERSION") -Raw).Trim()
$version = if ($raw -match '^v?(\d+\.\d+\.\d+)') { $Matches[1] } else { "0.0.0" }
if ($env:GITHUB_OUTPUT) { "version=$version" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8 }
Write-Host "Staged SubBuild Studio $version in $app"
Write-Output $version
