<#
.SYNOPSIS
    Download and unpack the TEXMEX SIFT1M (or siftsmall 10k) dataset into data/.

.DESCRIPTION
    Downloads sift.tar.gz (~160.5 MiB) or siftsmall.tar.gz (~5.06 MiB) from
    ftp://ftp.irisa.fr/local/texmex/corpus/ using Windows' built-in curl.exe
    (PowerShell's Invoke-WebRequest does not support ftp://) and unpacks it
    with tar.exe into data/.

.PARAMETER Small
    Download siftsmall.tar.gz (10k base vectors, 100 queries) instead of full SIFT1M.
#>
param(
    [switch]$Small
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $RepoRoot "data"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

if ($Small) {
    $ArchiveName = "siftsmall.tar.gz"
    $ExpectedDir = "siftsmall"
    $CheckFile = "siftsmall_base.fvecs"
} else {
    $ArchiveName = "sift.tar.gz"
    $ExpectedDir = "sift"
    $CheckFile = "sift_base.fvecs"
}

$Url = "ftp://ftp.irisa.fr/local/texmex/corpus/$ArchiveName"
$ArchivePath = Join-Path $DataDir $ArchiveName
$TargetCheck = Join-Path (Join-Path $DataDir $ExpectedDir) $CheckFile

if (Test-Path $TargetCheck) {
    Write-Host "Dataset already present at $TargetCheck"
    exit 0
}

Write-Host "Downloading $Url -> $ArchivePath ..."
& curl.exe -fL --retry 3 --retry-delay 2 -o $ArchivePath $Url
if ($LASTEXITCODE -ne 0) {
    Write-Error @"
Failed to download $Url (curl exit code $LASTEXITCODE).
If FTP is blocked on your network, download $ArchiveName manually from:
  $Url
Place it at:
  $ArchivePath
Then extract it into $DataDir using:
  tar -xzf "$ArchivePath" -C "$DataDir"
"@
    exit 1
}

Write-Host "Extracting $ArchivePath into $DataDir ..."
& tar.exe -xzf $ArchivePath -C $DataDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "tar.exe failed with exit code $LASTEXITCODE"
    exit 1
}

Remove-Item -Force $ArchivePath -ErrorAction SilentlyContinue
Write-Host "Done. Unpacked into $(Join-Path $DataDir $ExpectedDir)"
