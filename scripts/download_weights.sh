#!/usr/bin/env bash
# Download ViTPose-H (too large for GitHub LFS). Other weights ship in the repo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$ROOT/checkpoints"
NAME="td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
DEST="$DEST_DIR/$NAME"
URL="https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/$NAME"

mkdir -p "$DEST_DIR"
if [ -f "$DEST" ]; then
  echo "Already present: $DEST"
  exit 0
fi

echo "Downloading ViTPose-H (~2.4 GB) to $DEST"
curl -L --fail --retry 3 -o "$DEST" "$URL"
echo "Done."
