"""
Resolve bundled MMPose tree under app/mmpose (not ~/mmpose).

Override for custom layouts:
  MMPOSE_ROOT — root of the mmpose checkout (default: <app>/mmpose)
  FENCING18_CHECKPOINT — explicit fencing-18 ViTPose checkpoint (default: latest best*.pth in work_dir)
  FENCING18_WORK_DIR — directory with fencing-18 training checkpoints
  CHECKPOINTS_DIR — bundled OpenMMLab weights (default: <app>/checkpoints)
  RTMDET_CHECKPOINT — RTMDet person detector .pth (optional)
  MOTIONBERT_CHECKPOINT — MotionBERT 3D lift .pth (optional)
  MODEL_PATH — touch classifier v3.46 weights (default: best_touch_v346_coco17_bs10_multivid_val.pth)
  YOLO11S_ONNX / RTMPOSE_S_ONNX — arm-attempt ONNX stack (default under checkpoints/)
  ARM_ATTEMPT_BACKEND — onnx (default) | mmpose
"""

from __future__ import annotations

import os

_APP_DIR = os.path.dirname(os.path.abspath(__file__))

RTMDET_PERSON_CKPT_NAME = "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
RTMDET_PERSON_CKPT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
)

RTMPOSE_S_CKPT_NAME = (
    "rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth"
)
RTMPOSE_S_CKPT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth"
)

YOLO11S_ONNX_NAME = "yolo11s.onnx"
RTMPOSE_S_ONNX_NAME = "rtmpose_s_aic_coco_256x192.onnx"

MOTIONBERT_FT_CKPT_NAME = "motionbert_ft_h36m-d80af323_20230531.pth"
MOTIONBERT_FT_CKPT_URL = (
    "https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/"
    "motionbert_ft_h36m-d80af323_20230531.pth"
)

VITPOSE_H_CKPT_NAME = (
    "td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
)
VITPOSE_H_CKPT_URL = (
    "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
)

TOUCH_V346_CKPT_NAME = "best_touch_v346_coco17_bs10_multivid_val.pth"
ATTACK_MODEL_CKPT_NAME = "best_attack_3d_proximity_winrobust.pth"


def checkpoints_dir() -> str:
    return os.environ.get("CHECKPOINTS_DIR", os.path.join(_APP_DIR, "checkpoints"))


def _resolve_checkpoint(explicit_env: str, local_name: str, url: str) -> str:
    explicit = os.environ.get(explicit_env, "").strip()
    if explicit:
        return explicit
    local = os.path.join(checkpoints_dir(), local_name)
    if os.path.isfile(local):
        return local
    return url


def rtmdet_person_checkpoint_path() -> str:
    return _resolve_checkpoint(
        "RTMDET_CHECKPOINT", RTMDET_PERSON_CKPT_NAME, RTMDET_PERSON_CKPT_URL
    )


def motionbert_checkpoint_path() -> str:
    return _resolve_checkpoint(
        "MOTIONBERT_CHECKPOINT", MOTIONBERT_FT_CKPT_NAME, MOTIONBERT_FT_CKPT_URL
    )


def mmpose_root() -> str:
    return os.environ.get("MMPOSE_ROOT", os.path.join(_APP_DIR, "mmpose"))


def vitpose_coco17_config_path() -> str:
    """COCO-17 ViTPose-H config (v3.46 default pose stack)."""
    return os.path.join(_APP_DIR, "mmpose_configs", "vitpose_huge_coco17_256x192.py")


def vitpose_h_config_path() -> str:
    return vitpose_coco17_config_path()


def touch_classifier_default_path() -> str:
    explicit = os.environ.get("MODEL_PATH", "").strip()
    if explicit:
        return explicit
    return os.path.join(_APP_DIR, TOUCH_V346_CKPT_NAME)


def attack_classifier_default_path() -> str:
    explicit = os.environ.get("ATTACK_MODEL_PATH", "").strip()
    if explicit:
        return explicit
    return os.path.join(_APP_DIR, ATTACK_MODEL_CKPT_NAME)


def vitpose_h_checkpoint_path() -> str:
    return _resolve_checkpoint(
        "VITPOSE_H_CHECKPOINT", VITPOSE_H_CKPT_NAME, VITPOSE_H_CKPT_URL
    )


def fencing18_vitpose_work_dir() -> str:
    """Custom 18-kpt ViTPose work dir (training checkpoints)."""
    explicit = os.environ.get("FENCING18_WORK_DIR", "").strip()
    if explicit:
        return explicit
    candidates = [
        os.path.join(_APP_DIR, "work_dirs", "vitpose_huge_fencing18_256x192_stable17"),
        os.path.join(
            mmpose_root(),
            "work_dirs",
            "vitpose_huge_fencing18_256x192_stable17",
        ),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def fencing18_config_path() -> str:
    return os.path.join(_APP_DIR, "mmpose_configs", "vitpose_huge_fencing18_256x192.py")


def fencing18_metainfo_path() -> str:
    return os.path.join(_APP_DIR, "mmpose_configs", "fencing18_metainfo.py")


def fencing18_checkpoint_path() -> str:
    """Fencing-18 ViTPose checkpoint (incl. bellguard). Set FENCING18_CHECKPOINT or use work_dir best*.pth."""
    explicit = os.environ.get("FENCING18_CHECKPOINT", "").strip()
    if explicit:
        return explicit
    work_dir = fencing18_vitpose_work_dir()
    if os.path.isdir(work_dir):
        import glob

        best = glob.glob(os.path.join(work_dir, "best*.pth"))
        if best:
            return max(best, key=os.path.getmtime)
        epochs = glob.glob(os.path.join(work_dir, "epoch_*.pth"))
        if epochs:
            return max(epochs, key=os.path.getmtime)
    return ""


def rtmdet_person_config_path() -> str:
    return os.path.join(
        mmpose_root(),
        "demo",
        "mmdetection_cfg",
        "rtmdet_m_640-8xb32_coco-person.py",
    )


def rtmpose_s_config_path() -> str:
    explicit = os.environ.get("RTMPOSE_S_CONFIG", "").strip()
    if explicit:
        return explicit
    return os.path.join(
        mmpose_root(),
        "configs",
        "body_2d_keypoint",
        "rtmpose",
        "coco",
        "rtmpose-s_8xb256-420e_coco-256x192.py",
    )


def rtmpose_s_checkpoint_path() -> str:
    return _resolve_checkpoint(
        "RTMPOSE_S_CHECKPOINT", RTMPOSE_S_CKPT_NAME, RTMPOSE_S_CKPT_URL
    )


def _resolve_local_file(explicit_env: str, local_name: str) -> str:
    explicit = os.environ.get(explicit_env, "").strip()
    if explicit:
        return explicit
    return os.path.join(checkpoints_dir(), local_name)


def yolo11s_onnx_path() -> str:
    return _resolve_local_file("YOLO11S_ONNX", YOLO11S_ONNX_NAME)


def rtmpose_s_onnx_path() -> str:
    return _resolve_local_file("RTMPOSE_S_ONNX", RTMPOSE_S_ONNX_NAME)


def arm_attempt_backend() -> str:
    """onnx (default) | mmpose — bout-wide arm-attempt pose stack."""
    raw = os.environ.get("ARM_ATTEMPT_BACKEND", "onnx").strip().lower()
    if raw in ("mmpose", "pytorch", "rtmdet"):
        return "mmpose"
    return "onnx"
