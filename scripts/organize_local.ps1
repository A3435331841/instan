param(
    [switch]$Apply,
    [string]$Root = 'D:\instan',
    [string]$Storage = 'D:\instan\grt360_storage'
)

$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($Root).TrimEnd('\')
$Storage = [IO.Path]::GetFullPath($Storage).TrimEnd('\')
$Repo = Join-Path $Root 'pano360'
$Stamp = '20260827'
$LogDir = Join-Path $Storage 'manifests'
$LogPath = Join-Path $LogDir "LOCAL_REORGANIZATION_LOG_$Stamp.csv"

function Assert-SafePath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    if (-not $full.StartsWith($Root + '\', [StringComparison]::OrdinalIgnoreCase) -and
        $full -ne $Root) {
        throw "Refusing path outside workspace root: $full"
    }
    return $full
}

function Add-Action([string]$Source, [string]$Destination, [string]$Kind, [bool]$Junction) {
    $sourceFull = Assert-SafePath $Source
    $destinationFull = Assert-SafePath $Destination
    $script:Actions += [pscustomobject]@{
        source = $sourceFull; destination = $destinationFull; kind = $Kind
        junction = $Junction; source_exists = (Test-Path -LiteralPath $sourceFull)
        destination_exists = (Test-Path -LiteralPath $destinationFull)
    }
}

function Move-WithJunction([string]$Source, [string]$Destination, [string]$Kind) {
    $sourceFull = Assert-SafePath $Source
    $destinationFull = Assert-SafePath $Destination
    if (-not (Test-Path -LiteralPath $sourceFull)) { return }
    if (Test-Path -LiteralPath $destinationFull) {
        throw "Destination already exists; refusing to merge automatically: $destinationFull"
    }
    $parent = Split-Path -Parent $destinationFull
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Move-Item -LiteralPath $sourceFull -Destination $destinationFull
    New-Item -ItemType Junction -Path $sourceFull -Target $destinationFull | Out-Null
    Add-Content -LiteralPath $LogPath -Value ('"{0}","{1}","{2}","junction"' -f $sourceFull,$destinationFull,$Kind)
}

function Move-Only([string]$Source, [string]$Destination, [string]$Kind) {
    $sourceFull = Assert-SafePath $Source
    $destinationFull = Assert-SafePath $Destination
    if (-not (Test-Path -LiteralPath $sourceFull)) { return }
    if (Test-Path -LiteralPath $destinationFull) {
        throw "Destination already exists; refusing to merge automatically: $destinationFull"
    }
    $parent = Split-Path -Parent $destinationFull
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Move-Item -LiteralPath $sourceFull -Destination $destinationFull
    Add-Content -LiteralPath $LogPath -Value ('"{0}","{1}","{2}","move"' -f $sourceFull,$destinationFull,$Kind)
}

$script:Actions = @()

# Root-level items: preserve old paths for data and legacy deliverables.
Add-Action (Join-Path $Root '初赛数据') (Join-Path $Storage 'datasets\official_train') 'official_train' $true
Add-Action (Join-Path $Root 'deliverables') (Join-Path $Root 'grt360_deliverables\current') 'deliverables_current' $true
Add-Action (Join-Path $Root '交付物_2026-08-14') (Join-Path $Root 'grt360_deliverables\legacy_20260814') 'deliverables_legacy' $true
Add-Action (Join-Path $Root 'external') (Join-Path $Storage 'upstream_sources\legacy_external') 'external' $true

foreach ($name in @('downloads','smoke_dataset','smoke_dataset2','smoke_result','_docx_media','_docx_render','graphify-out','handoff_read_20260810','project')) {
    Add-Action (Join-Path $Root $name) (Join-Path $Root "grt360_scratch\$name") 'scratch' $false
}

# Ignored repository payloads: move them out of the Git working tree but keep
# the old paths as Junctions so existing scripts remain runnable.
Add-Action (Join-Path $Repo 'artifacts') (Join-Path $Storage 'experiments\legacy_artifacts') 'repo_artifacts' $true
Add-Action (Join-Path $Repo 'tools_local\uetrack_docker') (Join-Path $Storage 'docker_images\uetrack_context') 'docker_context' $true
Add-Action (Join-Path $Repo '.codex_tmp') (Join-Path $Root 'grt360_scratch\temp_kits') 'codex_temp' $false

# data360 is retained as metadata/splits; only ignored payloads are relocated.
foreach ($name in @('0001','0002','0003','0004','0005','zips','.cache')) {
    Add-Action (Join-Path $Repo "data360\$name") (Join-Path $Storage "datasets\360vot_legacy\$name") 'data360_payload' $true
}

if (-not $Apply) {
    $Actions | Format-Table -AutoSize
    Write-Output "DRY RUN ONLY. Re-run with -Apply after server archive and SHA256 verification."
    exit 0
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (-not (Test-Path -LiteralPath $LogPath)) {
    Set-Content -LiteralPath $LogPath -Value 'source,destination,kind,operation'
}

# Never move a path with unresolved conflicts.  The only deletion-like action
# in this script is creating an empty replacement directory for .codex_tmp.
foreach ($action in $Actions) {
    if ($action.destination_exists) {
        throw "Preflight conflict at destination: $($action.destination)"
    }
}

Move-WithJunction (Join-Path $Root '初赛数据') (Join-Path $Storage 'datasets\official_train') 'official_train'
Move-WithJunction (Join-Path $Root 'deliverables') (Join-Path $Root 'grt360_deliverables\current') 'deliverables_current'
Move-WithJunction (Join-Path $Root '交付物_2026-08-14') (Join-Path $Root 'grt360_deliverables\legacy_20260814') 'deliverables_legacy'
Move-WithJunction (Join-Path $Root 'external') (Join-Path $Storage 'upstream_sources\legacy_external') 'external'

foreach ($name in @('downloads','smoke_dataset','smoke_dataset2','smoke_result','_docx_media','_docx_render','graphify-out','handoff_read_20260810','project')) {
    Move-Only (Join-Path $Root $name) (Join-Path $Root "grt360_scratch\$name") 'scratch'
}

Move-WithJunction (Join-Path $Repo 'artifacts') (Join-Path $Storage 'experiments\legacy_artifacts') 'repo_artifacts'
Move-WithJunction (Join-Path $Repo 'tools_local\uetrack_docker') (Join-Path $Storage 'docker_images\uetrack_context') 'docker_context'
Move-Only (Join-Path $Repo '.codex_tmp') (Join-Path $Root 'grt360_scratch\temp_kits') 'codex_temp'
New-Item -ItemType Directory -Force -Path (Join-Path $Repo '.codex_tmp') | Out-Null

foreach ($name in @('0001','0002','0003','0004','0005','zips','.cache')) {
    Move-WithJunction (Join-Path $Repo "data360\$name") (Join-Path $Storage "datasets\360vot_legacy\$name") 'data360_payload'
}

Write-Output "Local organization complete. Log: $LogPath"
