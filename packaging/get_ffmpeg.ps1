# Downloads a static FFmpeg build (ffmpeg.exe + ffprobe.exe) into packaging\ffmpeg for the exe.
# Source: gyan.dev "release essentials" build. Run once; build_exe.bat skips it when present.
#
# The binaries ship inside the exe and parse every video the user opens, so they are checked
# against the SHA-256 hashes pinned in packaging\ffmpeg.sha256 (the build that was reviewed).
# When gyan.dev publishes a newer release the hashes no longer match and the script stops.
# To move to the new build: check its version and the hash gyan.dev publishes for the archive,
# then run   powershell -File packaging\get_ffmpeg.ps1 -AcceptNewBuild   to pin it.
param([switch]$AcceptNewBuild)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes Invoke-WebRequest very slow

$target = Join-Path $PSScriptRoot 'ffmpeg'
$url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
$zip = Join-Path $env:TEMP 'subbuild-ffmpeg.zip'
$unpacked = Join-Path $env:TEMP 'subbuild-ffmpeg'

Write-Host "Downloading FFmpeg from $url ..."
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
if (Test-Path $unpacked) { Remove-Item $unpacked -Recurse -Force }
Expand-Archive -Path $zip -DestinationPath $unpacked

$pinFile = Join-Path $PSScriptRoot 'ffmpeg.sha256'
$pinned = @{}
if (Test-Path $pinFile) {
    foreach ($line in Get-Content $pinFile) {
        $parts = $line.Trim() -split '\s+'
        if ($parts.Count -ge 2) { $pinned[$parts[1]] = $parts[0].ToLower() }
    }
}

$found = @{}
foreach ($name in 'ffmpeg.exe', 'ffprobe.exe') {
    $file = Get-ChildItem -Path $unpacked -Recurse -Filter $name | Select-Object -First 1
    if (-not $file) { throw "$name not found in the FFmpeg archive" }
    $found[$name] = $file
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower()
    if (-not $AcceptNewBuild -and $pinned[$name] -ne $hash) {
        Remove-Item $zip -Force
        Remove-Item $unpacked -Recurse -Force
        throw ("$name does not match the hash pinned in packaging\ffmpeg.sha256 (got $hash).`n" +
               "This is either a newer FFmpeg release or a tampered download. Check the version and the hash " +
               "published on gyan.dev, then re-run with -AcceptNewBuild to pin it.")
    }
    $found[$name + '.hash'] = $hash
}
if ($AcceptNewBuild) {
    "$($found['ffmpeg.exe.hash'])  ffmpeg.exe`n$($found['ffprobe.exe.hash'])  ffprobe.exe" |
        Set-Content -LiteralPath $pinFile -Encoding ascii
    Write-Host "Pinned the new FFmpeg build in $pinFile - commit it together with the version you reviewed."
}

New-Item -ItemType Directory -Force -Path $target | Out-Null
foreach ($name in 'ffmpeg.exe', 'ffprobe.exe') {
    Copy-Item $found[$name].FullName -Destination $target -Force
}
$license = Get-ChildItem -Path $unpacked -Recurse -Filter 'LICENSE*' | Select-Object -First 1
if ($license) { Copy-Item $license.FullName -Destination (Join-Path $target 'LICENSE.txt') -Force }

Remove-Item $zip -Force
Remove-Item $unpacked -Recurse -Force
Write-Host "FFmpeg ready in $target"
