#!/usr/bin/env bash
# Three-image build: ML base (rare) + custom weights (rare) + app (frequent).
# Weights image: COCO-17 ViTPose-H + ONNX YOLO11s/RTMPose-s + touch v3.46
#   (see prepare-weights-context.sh).
#
# After adding/changing ONNX arm-attempt weights, rebuild weights once:
#   BUILD_BASE=0 BUILD_WEIGHTS=1 ./scripts/build-docker.sh
# Day-to-day app-only rebuilds can keep BUILD_WEIGHTS=0.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"

PLATFORM="${PLATFORM:-linux/amd64}"
BASE_IMAGE="${BASE_IMAGE:-YOUR_DOCKERHUB/fencing-base:latest}"
WEIGHTS_IMAGE="${WEIGHTS_IMAGE:-YOUR_DOCKERHUB/fencing-weights:latest}"
APP_IMAGE="${APP_IMAGE:-YOUR_DOCKERHUB/fencing-mmpose:latest}"
PUSH="${PUSH:-1}"
BUILD_BASE="${BUILD_BASE:-1}"
BUILD_WEIGHTS="${BUILD_WEIGHTS:-1}"

build_args=()
if [[ "$PUSH" == "1" ]]; then
  build_args+=(--push)
else
  build_args+=(--load)
fi

if [[ "$BUILD_BASE" == "1" ]]; then
  echo "==> Building ML base image: $BASE_IMAGE"
  docker buildx build \
    --platform "$PLATFORM" \
    -f Dockerfile.base \
    -t "$BASE_IMAGE" \
    "${build_args[@]}" \
    .
else
  echo "==> Skipping base build (BUILD_BASE=0); using $BASE_IMAGE from registry"
fi

if [[ "$BUILD_WEIGHTS" == "1" ]]; then
  echo "==> Preparing weights context"
  bash "$APP_DIR/scripts/prepare-weights-context.sh"

  echo "==> Building weights image: $WEIGHTS_IMAGE"
  docker buildx build \
    --platform "$PLATFORM" \
    -f Dockerfile.weights \
    -t "$WEIGHTS_IMAGE" \
    "${build_args[@]}" \
    weights-context
else
  echo "==> Skipping weights build (BUILD_WEIGHTS=0); using $WEIGHTS_IMAGE from registry"
fi

echo "==> Building app image: $APP_IMAGE"
echo "    base=$BASE_IMAGE weights=$WEIGHTS_IMAGE"
docker buildx build \
  --platform "$PLATFORM" \
  -f Dockerfile \
  --build-arg "FENCING_BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "FENCING_WEIGHTS_IMAGE=$WEIGHTS_IMAGE" \
  -t "$APP_IMAGE" \
  "${build_args[@]}" \
  .

echo "Done."
