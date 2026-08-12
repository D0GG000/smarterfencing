# Download ViTPose-H (too large for GitHub LFS). Other weights ship in the repo.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DestDir = Join-Path $Root "checkpoints"
$Name = "td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
$Dest = Join-Path $DestDir $Name
$Url = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/$Name"

New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
if (Test-Path $Dest) {
    Write-Host "Already present: $Dest"
    exit 0
}

Write-Host "Downloading ViTPose-H (~2.4 GB) to $Dest"
Invoke-WebRequest -Uri $Url -OutFile $Dest
Write-Host "Done."
