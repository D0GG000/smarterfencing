#!/usr/bin/env bash
# Stage custom model weights for Dockerfile.weights:
#   - COCO-17 ViTPose-H under checkpoints/
#   - ONNX YOLO11s-det + RTMPose-s under checkpoints/ (arm-attempt pass)
#   - touch classifier v3.46 under touch/
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DST="$APP_DIR/weights-context"

VITPOSE_CKPT_NAME="td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
VITPOSE_CKPT_URL="https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/${VITPOSE_CKPT_NAME}"
VITPOSE_LOCAL="$APP_DIR/checkpoints/$VITPOSE_CKPT_NAME"

YOLO11S_ONNX_NAME="yolo11s.onnx"
RTMPOSE_S_ONNX_NAME="rtmpose_s_aic_coco_256x192.onnx"
YOLO11S_ONNX_LOCAL="$APP_DIR/checkpoints/$YOLO11S_ONNX_NAME"
RTMPOSE_S_ONNX_LOCAL="$APP_DIR/checkpoints/$RTMPOSE_S_ONNX_NAME"

TOUCH_CKPT_NAME="best_touch_v346_coco17_bs10_multivid_val.pth"
TOUCH_LOCAL="$APP_DIR/$TOUCH_CKPT_NAME"
ATTACK_CKPT_NAME="best_attack_3d_proximity.pth"
ATTACK_LOCAL="$APP_DIR/$ATTACK_CKPT_NAME"

rm -rf "$DST"
mkdir -p "$DST/checkpoints" "$DST/touch" "$DST/attack"

resolve_vitpose_checkpoint() {
  if [[ -n "${VITPOSE_H_CHECKPOINT:-}" && -f "${VITPOSE_H_CHECKPOINT}" ]]; then
    echo "${VITPOSE_H_CHECKPOINT}"
    return 0
  fi
  if [[ -f "$VITPOSE_LOCAL" ]]; then
    echo "$VITPOSE_LOCAL"
    return 0
  fi
  return 1
}

resolve_touch_checkpoint() {
  if [[ -n "${TOUCH_MODEL_PATH:-}" && -f "${TOUCH_MODEL_PATH}" ]]; then
    echo "${TOUCH_MODEL_PATH}"
    return 0
  fi
  if [[ -n "${MODEL_PATH:-}" && -f "${MODEL_PATH}" ]]; then
    echo "${MODEL_PATH}"
    return 0
  fi
  if [[ -f "$TOUCH_LOCAL" ]]; then
    echo "$TOUCH_LOCAL"
    return 0
  fi
  return 1
}

resolve_attack_checkpoint() {
  if [[ -n "${ATTACK_MODEL_PATH:-}" && -f "${ATTACK_MODEL_PATH}" ]]; then
    echo "${ATTACK_MODEL_PATH}"
    return 0
  fi
  if [[ -f "$ATTACK_LOCAL" ]]; then
    echo "$ATTACK_LOCAL"
    return 0
  fi
  for alt in \
      "$APP_DIR/annotate_attack/$ATTACK_CKPT_NAME" \
      "$APP_DIR/best_attack_3d_angles.pth" \
      "$APP_DIR/best_attack_3d_fullseq.pth" \
      "$APP_DIR/best_attack_3d_vitpose_coco17.pth" \
      "$APP_DIR/best_attack_3d.pth"; do
    if [[ -f "$alt" ]]; then
      echo "$alt"
      return 0
    fi
  done
  return 1
}

vitpose_dest="$DST/checkpoints/$VITPOSE_CKPT_NAME"
touch_dest="$DST/touch/$TOUCH_CKPT_NAME"
attack_dest="$DST/attack/$ATTACK_CKPT_NAME"

if chosen="$(resolve_vitpose_checkpoint)"; then
  echo "Using COCO-17 ViTPose-H checkpoint: $chosen ($(du -h "$chosen" | awk '{print $1}'))"
  cp -f "$chosen" "$vitpose_dest"
else
  echo "COCO-17 ViTPose-H not found locally; downloading from OpenMMLab..."
  mkdir -p "$(dirname "$VITPOSE_LOCAL")"
  curl -fL --retry 3 --retry-delay 5 -o "$vitpose_dest" "$VITPOSE_CKPT_URL"
  echo "Downloaded ViTPose-H ($(du -h "$vitpose_dest" | awk '{print $1}'))"
fi

# Arm-attempt ONNX stack (required for default ARM_ATTEMPT_BACKEND=onnx).
yolo_onnx_dest="$DST/checkpoints/$YOLO11S_ONNX_NAME"
rtmpose_onnx_dest="$DST/checkpoints/$RTMPOSE_S_ONNX_NAME"
if [[ -n "${YOLO11S_ONNX:-}" && -f "${YOLO11S_ONNX}" ]]; then
  cp -f "${YOLO11S_ONNX}" "$yolo_onnx_dest"
elif [[ -f "$YOLO11S_ONNX_LOCAL" ]]; then
  cp -f "$YOLO11S_ONNX_LOCAL" "$yolo_onnx_dest"
else
  echo "ERROR: YOLO11s ONNX not found." >&2
  echo "  Place $YOLO11S_ONNX_LOCAL or set YOLO11S_ONNX" >&2
  exit 1
fi
echo "Using YOLO11s ONNX: $yolo_onnx_dest ($(du -h "$yolo_onnx_dest" | awk '{print $1}'))"

if [[ -n "${RTMPOSE_S_ONNX:-}" && -f "${RTMPOSE_S_ONNX}" ]]; then
  cp -f "${RTMPOSE_S_ONNX}" "$rtmpose_onnx_dest"
elif [[ -f "$RTMPOSE_S_ONNX_LOCAL" ]]; then
  cp -f "$RTMPOSE_S_ONNX_LOCAL" "$rtmpose_onnx_dest"
else
  echo "ERROR: RTMPose-s ONNX not found." >&2
  echo "  Place $RTMPOSE_S_ONNX_LOCAL or set RTMPOSE_S_ONNX" >&2
  exit 1
fi
echo "Using RTMPose-s ONNX: $rtmpose_onnx_dest ($(du -h "$rtmpose_onnx_dest" | awk '{print $1}'))"

if chosen="$(resolve_touch_checkpoint)"; then
  echo "Using touch v3.46 checkpoint: $chosen ($(du -h "$chosen" | awk '{print $1}'))"
  cp -f "$chosen" "$touch_dest"
else
  echo "ERROR: touch v3.46 checkpoint not found." >&2
  echo "  Set TOUCH_MODEL_PATH (or MODEL_PATH), or place:" >&2
  echo "  $TOUCH_LOCAL" >&2
  exit 1
fi

if chosen="$(resolve_attack_checkpoint)"; then
  echo "Using attack classifier checkpoint: $chosen ($(du -h "$chosen" | awk '{print $1}'))"
  cp -f "$chosen" "$attack_dest"
else
  echo "WARN: attack classifier checkpoint not found; attack-type predictions will be skipped at runtime." >&2
  echo "  Set ATTACK_MODEL_PATH or place $ATTACK_LOCAL" >&2
  echo "  (annotate_attack/best_attack_3d_proximity.pth from TrainingAttack3D.py)" >&2
  touch "$DST/attack/.keep"
fi

ODTRACK_CKPT_NAME="ODTrack_ep0300.pth.tar"
ODTRACK_LOCAL="$APP_DIR/models/odtrack/$ODTRACK_CKPT_NAME"
odtrack_dest="$DST/odtrack/$ODTRACK_CKPT_NAME"
mkdir -p "$DST/odtrack"

resolve_odtrack_checkpoint() {
  if [[ -n "${ODTRACK_CHECKPOINT:-}" && -f "${ODTRACK_CHECKPOINT}" ]]; then
    echo "${ODTRACK_CHECKPOINT}"
    return 0
  fi
  if [[ -f "$ODTRACK_LOCAL" ]]; then
    echo "$ODTRACK_LOCAL"
    return 0
  fi
  return 1
}

if chosen="$(resolve_odtrack_checkpoint)"; then
  echo "Using ODTrack checkpoint: $chosen ($(du -h "$chosen" | awk '{print $1}'))"
  cp -f "$chosen" "$odtrack_dest"
else
  echo "WARN: ODTrack checkpoint not found; tracking will download at runtime or fall back to legacy." >&2
  echo "  Run: python setup_odtrack.py  or set ODTRACK_CHECKPOINT" >&2
  touch "$DST/odtrack/.keep"
fi

echo "Prepared weights in $DST:"
find "$DST" -type f \( -name '*.pth' -o -name '*.pth.tar' -o -name '*.onnx' \) -exec ls -lh {} \;
