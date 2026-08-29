[CmdletBinding()]
param(
    [ValidateSet('ort', 'torch')]
    [string]$Backend = 'ort',
    [Parameter(Mandatory = $true)]
    [string]$Context,
    [Parameter(Mandatory = $true)]
    [string]$Tag,
    [string]$BaseImage = 'nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04'
)

$ErrorActionPreference = 'Stop'
$contextPath = (Resolve-Path -LiteralPath $Context).Path
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker is not available on PATH. This command builds locally only and never pushes.'
}
$dockerfile = Join-Path $contextPath ("src/docker/final/Dockerfile.{0}-cu128" -f $Backend)
if (-not (Test-Path -LiteralPath $dockerfile)) {
    throw "Delivery build context is incomplete: $dockerfile"
}
if (-not (Test-Path -LiteralPath (Join-Path $contextPath 'models'))) {
    throw "Delivery build context is missing models/: $contextPath"
}
if ($Backend -eq 'torch' -and -not (Test-Path -LiteralPath (Join-Path $contextPath 'sutrack_src'))) {
    throw "Torch delivery context is missing sutrack_src/: $contextPath"
}

Write-Host "Building $Backend image locally; no registry push will be performed."
& docker build --file $dockerfile --tag $Tag --build-arg "BASE_IMAGE=$BaseImage" $contextPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& docker image inspect $Tag --format '{{.Id}} {{.Size}}'
