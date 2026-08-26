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
$MapPath = Join-Path $LogDir "LOCAL_PATH_MAP_$Stamp.csv"
$InventoryPath = Join-Path $LogDir "LOCAL_SOURCE_INVENTORY_$Stamp.csv"

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
    $alreadyMigrated = $false
    if (Test-Path -LiteralPath $sourceFull) {
        $sourceItem = Get-Item -LiteralPath $sourceFull -Force
        if ($sourceItem.LinkType -eq 'Junction' -and
            ([IO.Path]::GetFullPath([string]$sourceItem.Target).TrimEnd('\') -ieq $destinationFull)) {
            $alreadyMigrated = $true
        }
    }
    $destinationOccupied = $false
    if ((-not $alreadyMigrated) -and (Test-Path -LiteralPath $destinationFull)) {
        $destinationItem = Get-Item -LiteralPath $destinationFull
        $destinationOccupied = (-not $destinationItem.PSIsContainer) -or
            ($null -ne (Get-ChildItem -LiteralPath $destinationFull -Force | Select-Object -First 1))
    }
    $script:Actions += [pscustomobject]@{
        source = $sourceFull; destination = $destinationFull; kind = $Kind
        junction = $Junction; source_exists = (Test-Path -LiteralPath $sourceFull)
        destination_exists = $destinationOccupied; already_migrated = $alreadyMigrated
    }
}

function Move-EmptySkeleton([string]$Destination) {
    if (-not (Test-Path -LiteralPath $Destination)) { return }
    $item = Get-Item -LiteralPath $Destination
    if (-not $item.PSIsContainer) { throw "Destination is a file: $Destination" }
    if ($null -ne (Get-ChildItem -LiteralPath $Destination -Force | Select-Object -First 1)) {
        throw "Destination is non-empty: $Destination"
    }
    $backup = "$Destination.preexisting_empty_$Stamp"
    if (Test-Path -LiteralPath $backup) { throw "Skeleton backup already exists: $backup" }
    Move-Item -LiteralPath $Destination -Destination $backup
    Add-Content -LiteralPath $LogPath -Value ('"{0}","{1}","skeleton","move"' -f $Destination,$backup)
}

function Move-WithJunction([string]$Source, [string]$Destination, [string]$Kind) {
    $sourceFull = Assert-SafePath $Source
    $destinationFull = Assert-SafePath $Destination
    if (-not (Test-Path -LiteralPath $sourceFull)) { return }
    $sourceItem = Get-Item -LiteralPath $sourceFull -Force
    if ($sourceItem.LinkType -eq 'Junction' -and
        ([IO.Path]::GetFullPath([string]$sourceItem.Target).TrimEnd('\') -ieq $destinationFull)) { return }
    if (Test-Path -LiteralPath $destinationFull) { Move-EmptySkeleton $destinationFull }
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
    if ($Kind -eq 'codex_temp' -and (Test-Path -LiteralPath $destinationFull)) {
        $sourceItem = Get-Item -LiteralPath $sourceFull -Force
        if ($sourceItem.PSIsContainer -and $null -eq (Get-ChildItem -LiteralPath $sourceFull -Force | Select-Object -First 1)) {
            return
        }
    }
    if (Test-Path -LiteralPath $destinationFull) { Move-EmptySkeleton $destinationFull }
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

# Loose root documents/media are kept, but moved out of the workspace root.
foreach ($name in @('比赛策略_影石全景跟踪赛道.docx','比赛策略_影石全景跟踪赛道.md','全景视频智能跟踪赛道 (7.13更新).docx','_transcript_read.txt')) {
    Add-Action (Join-Path $Root $name) (Join-Path $Root ('grt360_deliverables\reference_docs\' + $name)) 'reference_doc' $false
}
foreach ($name in @('airsim_index.html','modlens_test.png','cockpit-provider-model-catalog.json','imagenet_classes.txt')) {
    Add-Action (Join-Path $Root $name) (Join-Path $Root "grt360_scratch\misc\$name") 'misc' $false
}
Add-Action (Join-Path $Root 'archive_listing.txt') (Join-Path $Storage 'manifests\archive_listing_20260827.txt') 'audit_snapshot' $false
Add-Action (Join-Path $Root '.qa_grt360_appendix_20260809') (Join-Path $Root 'grt360_deliverables\legacy_20260809') 'qa_archive' $false
Add-Action (Join-Path $Root 'tools_local') (Join-Path $Storage 'experiments\local_legacy_202608\tools_local') 'root_tools' $true
Add-Action (Join-Path $Repo 'docker\sutrack\build_ctx') (Join-Path $Storage 'docker_images\sutrack_build_ctx') 'docker_build_context' $false
Add-Action (Join-Path $Repo 'scripts\profile_sutrack.py') (Join-Path $Storage 'experiments\local_legacy_202608\profile_sutrack.py') 'profile_snapshot' $false

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
# Snapshot the planned map and source directory statistics before any move.
# These are audit artifacts; they intentionally contain no file contents or
# credentials and make the operation reversible through the log/Junctions.
$Actions | Export-Csv -LiteralPath $MapPath -NoTypeInformation -Encoding UTF8
$inventory = foreach ($action in $Actions) {
    if (-not (Test-Path -LiteralPath $action.source)) { continue }
    $files = @(Get-ChildItem -LiteralPath $action.source -Recurse -File -Force -ErrorAction SilentlyContinue)
    [pscustomobject]@{
        source = $action.source
        destination = $action.destination
        kind = $action.kind
        files = $files.Count
        bytes = (($files | Measure-Object -Property Length -Sum).Sum -as [long])
    }
}
$inventory | Export-Csv -LiteralPath $InventoryPath -NoTypeInformation -Encoding UTF8

# Never move a path with unresolved conflicts.  The only deletion-like action
# in this script is creating an empty replacement directory for .codex_tmp.
foreach ($action in $Actions) {
    $emptyCodexSource = $false
    if ($action.kind -eq 'codex_temp' -and $action.source_exists) {
        $sourceItem = Get-Item -LiteralPath $action.source -Force
        $emptyCodexSource = $sourceItem.PSIsContainer -and
            ($null -eq (Get-ChildItem -LiteralPath $action.source -Force | Select-Object -First 1))
    }
    if ($action.destination_exists -and $action.source_exists -and -not $action.already_migrated -and -not $emptyCodexSource) {
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

foreach ($name in @('比赛策略_影石全景跟踪赛道.docx','比赛策略_影石全景跟踪赛道.md','全景视频智能跟踪赛道 (7.13更新).docx','_transcript_read.txt')) {
    Move-Only (Join-Path $Root $name) (Join-Path $Root ('grt360_deliverables\reference_docs\' + $name)) 'reference_doc'
}
foreach ($name in @('airsim_index.html','modlens_test.png','cockpit-provider-model-catalog.json','imagenet_classes.txt')) {
    Move-Only (Join-Path $Root $name) (Join-Path $Root "grt360_scratch\misc\$name") 'misc'
}
Move-Only (Join-Path $Root 'archive_listing.txt') (Join-Path $Storage 'manifests\archive_listing_20260827.txt') 'audit_snapshot'
Move-Only (Join-Path $Root '.qa_grt360_appendix_20260809') (Join-Path $Root 'grt360_deliverables\legacy_20260809') 'qa_archive'
Move-WithJunction (Join-Path $Root 'tools_local') (Join-Path $Storage 'experiments\local_legacy_202608\tools_local') 'root_tools'
Move-Only (Join-Path $Repo 'docker\sutrack\build_ctx') (Join-Path $Storage 'docker_images\sutrack_build_ctx') 'docker_build_context'
Move-Only (Join-Path $Repo 'scripts\profile_sutrack.py') (Join-Path $Storage 'experiments\local_legacy_202608\profile_sutrack.py') 'profile_snapshot'

Move-WithJunction (Join-Path $Repo 'artifacts') (Join-Path $Storage 'experiments\legacy_artifacts') 'repo_artifacts'
Move-WithJunction (Join-Path $Repo 'tools_local\uetrack_docker') (Join-Path $Storage 'docker_images\uetrack_context') 'docker_context'
Move-Only (Join-Path $Repo '.codex_tmp') (Join-Path $Root 'grt360_scratch\temp_kits') 'codex_temp'
New-Item -ItemType Directory -Force -Path (Join-Path $Repo '.codex_tmp') | Out-Null

foreach ($name in @('0001','0002','0003','0004','0005','zips','.cache')) {
    Move-WithJunction (Join-Path $Repo "data360\$name") (Join-Path $Storage "datasets\360vot_legacy\$name") 'data360_payload'
}

Write-Output "Local organization complete. Log: $LogPath; map: $MapPath; inventory: $InventoryPath"
