"""
Run inference with your trained 18-keypoint ViTPose checkpoint (MMPose).

- Uses **MMDet RTMDet** for person boxes, then **expands** them (same idea as
  `expand_coco_bboxes.py`: scale about center + optional padding) before pose,
  so crops match training better than raw detector boxes.
- Uses **raw** images by default (`fullframes/`), not `vitpose_fullframes_output/`
  (those JPEGs already have pose overlays — running inference on them causes
  “double” drawings).

Training note: COCO still lists **every** person RTMDet+ViTPose found; only the
**two largest** boxes per image get a labeled bellguard (18th keypoint). Other
people still contribute **body** supervision. By default this script keeps only
the **two largest detector boxes** per image (`--top-k-persons 2`) so inference
matches that fencer selection; use `--top-k-persons 0` to run pose on everyone
RTMDet finds.

Examples:
  python test_fencing_vitpose18.py
  python test_fencing_vitpose18.py --split train
  python test_fencing_vitpose18.py --split all
  python test_fencing_vitpose18.py --checkpoint "C:\\Users\\jorda\\mmpose\\work_dirs\\...\\best_coco_AP_epoch_10.pth"
  python test_fencing_vitpose18.py --max-images 20 --bbox-scale 1.2 --bbox-pad 32

Requires: mmpose-env with mmdet installed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional

import numpy as np

from mmpose_paths import fencing18_vitpose_work_dir, mmpose_root

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Same order as fencing18_metainfo / COCO + bellguard (index 17 = bellguard)
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "bellguard",
]

DEFAULT_DET_CKPT = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
)


def default_det_checkpoint() -> str:
    from mmpose_paths import rtmdet_person_checkpoint_path

    return rtmdet_person_checkpoint_path()


def normalize_coco_file_name(file_name: str) -> str:
    return os.path.basename(file_name.replace("\\", "/"))


def paths_from_coco_ann(coco_json: str, image_dir: str) -> List[str]:
    """Image paths in COCO `images` order (by id); only files that exist on disk."""
    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)
    rows = sorted(coco.get("images") or [], key=lambda im: int(im["id"]))
    paths: List[str] = []
    missing = 0
    for im in rows:
        fn = normalize_coco_file_name(im["file_name"])
        p = os.path.join(image_dir, fn)
        if os.path.isfile(p):
            paths.append(p)
        else:
            missing += 1
            print(f"  warning: COCO lists {fn!r} but file missing under {image_dir}")
    if missing:
        print(f"  ({missing} image(s) from COCO not found on disk)")
    return paths


def list_images(folder: str, max_images: Optional[int]) -> List[str]:
    out = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            out.append(path)
    if max_images is not None:
        out = out[: max_images]
    return out


def pick_latest_checkpoint(work_dir: str) -> Optional[str]:
    if not os.path.isdir(work_dir):
        return None
    best = glob.glob(os.path.join(work_dir, "best*.pth"))
    if best:
        return max(best, key=os.path.getmtime)
    epochs = glob.glob(os.path.join(work_dir, "epoch_*.pth"))
    if epochs:
        return max(epochs, key=os.path.getmtime)
    return None


def expand_bboxes_xyxy(
    bboxes: np.ndarray,
    img_h: int,
    img_w: int,
    scale: float,
    pad_px: float,
) -> np.ndarray:
    """Expand xyxy boxes: uniform pad, then scale about center, clip to image."""
    if bboxes.size == 0:
        return bboxes
    out = []
    for row in bboxes:
        x1, y1, x2, y2 = map(float, row[:4])
        x1 -= pad_px
        y1 -= pad_px
        x2 += pad_px
        y2 += pad_px
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        nw = w * scale
        nh = h * scale
        nx1 = cx - nw / 2.0
        ny1 = cy - nh / 2.0
        nx2 = cx + nw / 2.0
        ny2 = cy + nh / 2.0
        nx1 = max(0.0, min(nx1, float(img_w)))
        ny1 = max(0.0, min(ny1, float(img_h)))
        nx2 = max(0.0, min(nx2, float(img_w)))
        ny2 = max(0.0, min(ny2, float(img_h)))
        if nx2 <= nx1 or ny2 <= ny1:
            out.append(row[:4])
        else:
            out.append([nx1, ny1, nx2, ny2])
    return np.array(out, dtype=np.float32)


def filter_topk_by_area_xyxy(
    bboxes: np.ndarray,
    k: int,
    order_left_to_right: bool,
) -> np.ndarray:
    """Keep k largest boxes by area; optionally order by center-x (left fencer first)."""
    if bboxes.size == 0 or k <= 0:
        return bboxes
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-areas)
    order = order[: min(k, len(order))]
    sel = bboxes[order].copy()
    if order_left_to_right and len(sel) > 1:
        cx = (sel[:, 0] + sel[:, 2]) / 2.0
        sel = sel[np.argsort(cx)]
    return sel


def scores_payload_from_merged(name: str, data_samples) -> dict:
    """Build JSON-serializable dict of per-instance keypoint scores (incl. bellguard)."""
    import torch

    base = {"file_name": name, "instances": []}
    if data_samples is None or not hasattr(data_samples, "pred_instances"):
        return base
    inst = data_samples.pred_instances
    if inst is None:
        return base
    try:
        n = len(inst)
    except Exception:
        return base
    if n == 0:
        return base

    kpts = inst.keypoints
    ksc = inst.keypoint_scores
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.detach().cpu().numpy()
    if isinstance(ksc, torch.Tensor):
        ksc = ksc.detach().cpu().numpy()

    nk = ksc.shape[1] if ksc.ndim >= 2 else 0
    if nk < 18:
        base["error"] = f"expected 18 keypoint scores, got {nk}"
        return base

    bbox_scores = None
    if hasattr(inst, "bbox_scores") and inst.bbox_scores is not None:
        bs = inst.bbox_scores
        bbox_scores = bs.detach().cpu().numpy() if isinstance(bs, torch.Tensor) else np.asarray(bs)

    labels = ["left", "right"] if n == 2 else [f"person_{j}" for j in range(n)]
    out_insts = []
    for i in range(n):
        scores_row = [float(x) for x in ksc[i].tolist()]
        xy = kpts[i, 17].tolist()
        bell = {
            "x": float(xy[0]),
            "y": float(xy[1]),
            "score": float(ksc[i, 17]),
        }
        by_name = {KEYPOINT_NAMES[j]: scores_row[j] for j in range(min(len(KEYPOINT_NAMES), len(scores_row)))}
        out_insts.append(
            {
                "index": i,
                "label": labels[i] if i < len(labels) else f"person_{i}",
                "bbox_score": float(bbox_scores[i]) if bbox_scores is not None else None,
                "bellguard": bell,
                "keypoint_scores": scores_row,
                "keypoint_scores_by_name": by_name,
            }
        )
    base["instances"] = out_insts
    return base


def process_one_image(
    img_path: str,
    img_bgr: np.ndarray,
    detector,
    pose_estimator,
    visualizer,
    args,
):
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    from mmpose.evaluation.functional import nms
    from mmpose.structures import merge_data_samples

    import mmcv

    name = os.path.basename(img_path)

    det_result = inference_detector(detector, img_bgr)
    pred_instance = det_result.pred_instances.cpu().numpy()
    dets = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    dets = dets[
        np.logical_and(
            pred_instance.labels == args.det_cat_id,
            pred_instance.scores > args.bbox_thr,
        )
    ]
    dets = dets[nms(dets, args.nms_thr), :4]
    dets = filter_topk_by_area_xyxy(
        dets,
        args.top_k_persons,
        args.order_fencers_lr,
    )

    if dets.size == 0:
        payload = {"file_name": name, "instances": [], "note": "no person detections after NMS/top-k"}
        img_rgb = mmcv.bgr2rgb(img_bgr)
        return mmcv.rgb2bgr(img_rgb), payload

    h, w = img_bgr.shape[:2]
    if args.no_expand:
        bboxes = dets
    else:
        bboxes = expand_bboxes_xyxy(dets, h, w, args.bbox_scale, args.bbox_pad)

    pose_results = inference_topdown(pose_estimator, img_bgr, bboxes)
    data_samples = merge_data_samples(pose_results)
    payload = scores_payload_from_merged(name, data_samples)

    img_rgb = mmcv.bgr2rgb(img_bgr)
    visualizer.add_datasample(
        "result",
        img_rgb,
        data_sample=data_samples,
        draw_gt=False,
        draw_bbox=args.draw_bbox,
        show=False,
        kpt_thr=args.kpt_thr,
    )
    return mmcv.rgb2bgr(visualizer.get_image()), payload


def main() -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(base, "mmpose_configs", "vitpose_huge_fencing18_256x192.py")
    default_work = fencing18_vitpose_work_dir()
    default_mmpose_root = mmpose_root()
    default_det_config = os.path.join(
        default_mmpose_root, "demo", "mmdetection_cfg", "rtmdet_m_640-8xb32_coco-person.py"
    )
    default_input = os.path.join(base, "fullframes")
    default_out = os.path.join(base, "vitpose_fullframes_output", "test_infer_out")
    default_ann_dir = os.path.join(base, "vitpose_fullframes_output", "annotations")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=default_config, help="Pose training config (.py)")
    ap.add_argument("--checkpoint", default=None, help="Pose .pth (default: newest in work-dir)")
    ap.add_argument("--work-dir", default=default_work, help="Auto-pick pose checkpoint")
    ap.add_argument(
        "--mmpose-root",
        default=default_mmpose_root,
        help="MMPose repo (for default RTMDet config under demo/mmdetection_cfg)",
    )
    ap.add_argument(
        "--det-config",
        default=None,
        help="MMDet person config (default: <mmpose-root>/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py)",
    )
    ap.add_argument(
        "--det-checkpoint",
        default=None,
        help="RTMDet weights path or URL (default: bundled /app/checkpoints or URL)",
    )
    ap.add_argument(
        "--input",
        default=default_input,
        help="Raw image file or folder (default: fullframes — not pre-overlaid outputs)",
    )
    ap.add_argument(
        "--split",
        choices=("val", "train", "all"),
        default="val",
        help="val/train: only images listed in the COCO split JSON (default: val). "
        "all: every image file in --input folder.",
    )
    ap.add_argument(
        "--coco-ann",
        default=None,
        help="Override COCO JSON path (default: annotations/<split>_bbox_expand.json)",
    )
    ap.add_argument(
        "--ann-dir",
        default=default_ann_dir,
        help="Folder containing train_bbox_expand.json / val_bbox_expand.json",
    )
    ap.add_argument("--output-dir", default=default_out, help="Save visualizations here")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda:0 or cpu")
    ap.add_argument("--bbox-thr", type=float, default=0.3, dest="bbox_thr")
    ap.add_argument("--nms-thr", type=float, default=0.3, dest="nms_thr")
    ap.add_argument("--det-cat-id", type=int, default=0, dest="det_cat_id")
    ap.add_argument("--kpt-thr", type=float, default=0.3, dest="kpt_thr")
    ap.add_argument("--draw-bbox", action="store_true", default=True)
    ap.add_argument(
        "--bbox-scale",
        type=float,
        default=1.2,
        help="Match expand_coco_bboxes.py --scale (applied after --bbox-pad)",
    )
    ap.add_argument(
        "--bbox-pad",
        type=float,
        default=0.0,
        help="Extra pixels to grow each box before scaling (training used kpt-margin on GT bell; use e.g. 32 if weapon sticks out)",
    )
    ap.add_argument(
        "--no-expand",
        action="store_true",
        help="Use raw RTMDet boxes (no scale/pad)",
    )
    ap.add_argument(
        "--top-k-persons",
        type=int,
        default=2,
        help="After NMS, keep only the K largest person boxes by area (default: 2, "
        "same idea as merge script). Use 0 to keep all detections.",
    )
    ap.add_argument(
        "--no-order-lr",
        dest="order_fencers_lr",
        action="store_false",
        help="Do not sort the K boxes left-to-right by center x (default: sort)",
    )
    ap.set_defaults(order_fencers_lr=True)
    ap.add_argument(
        "--no-print-scores",
        dest="print_scores",
        action="store_false",
        help="Do not print bellguard scores to terminal",
    )
    ap.add_argument(
        "--no-save-scores-json",
        dest="save_scores_json",
        action="store_false",
        help="Do not write *_scores.json or all_scores.json",
    )
    ap.set_defaults(print_scores=True, save_scores_json=True)
    args = ap.parse_args()

    try:
        import mmcv
        import torch
        from mmdet.apis import init_detector
        from mmpose.apis import init_model as init_pose_estimator
        from mmpose.registry import VISUALIZERS
        from mmpose.utils import adapt_mmdet_pipeline
    except ImportError as e:
        print("Import error:", e)
        print("Activate mmpose-env (with mmdet) and retry.")
        return 1

    det_cfg = args.det_config or os.path.join(
        args.mmpose_root, "demo", "mmdetection_cfg", "rtmdet_m_640-8xb32_coco-person.py"
    )
    if not os.path.isfile(det_cfg):
        print(f"Detector config not found: {det_cfg}")
        print("Set --det-config or --mmpose-root to your MMPose clone.")
        return 1

    ckpt = args.checkpoint or pick_latest_checkpoint(args.work_dir)
    if not ckpt or not os.path.isfile(ckpt):
        print("Pose checkpoint not found. Pass --checkpoint or fix --work-dir.")
        print(f"  looked in: {args.work_dir}")
        return 1
    if not os.path.isfile(args.config):
        print(f"Pose config not found: {args.config}")
        return 1

    if os.path.isfile(args.input):
        paths = [args.input]
    elif os.path.isdir(args.input):
        if args.split == "all":
            paths = list_images(args.input, args.max_images)
        else:
            if args.coco_ann:
                ann_path = args.coco_ann
            else:
                name = f"{args.split}_bbox_expand.json"
                ann_path = os.path.join(args.ann_dir, name)
            if not os.path.isfile(ann_path):
                print(f"COCO split file not found: {ann_path}")
                print("Run expand_coco_bboxes.py or pass --coco-ann / fix --ann-dir.")
                return 1
            paths = paths_from_coco_ann(ann_path, args.input)
            if args.max_images is not None:
                paths = paths[: args.max_images]
    else:
        print(f"Input not found: {args.input}")
        return 1

    if not paths:
        print("No images to process.")
        return 1

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Pose config:     {args.config}")
    print(f"Pose weights:      {ckpt}")
    print(f"Det config:        {det_cfg}")
    print(
        f"Input (raw):       {args.input}  ({len(paths)} image(s), split={args.split})"
    )
    print(f"BBox expand:       scale={args.bbox_scale}, pad={args.bbox_pad}, no_expand={args.no_expand}")
    print(f"Output:            {args.output_dir}")
    print(f"Device:            {device}")

    det_ckpt = args.det_checkpoint or default_det_checkpoint()
    detector = init_detector(det_cfg, det_ckpt, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_pose_estimator(args.config, ckpt, device=device)

    pose_estimator.cfg.visualizer.radius = 4
    pose_estimator.cfg.visualizer.line_width = 2
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style="mmpose")

    combined_scores: List[dict] = []

    for i, img_path in enumerate(paths):
        name = os.path.basename(img_path)
        print(f"[{i + 1}/{len(paths)}] {name}")
        img_bgr = mmcv.imread(img_path)
        if img_bgr is None:
            print("  skip: could not read")
            continue
        out_bgr, payload = process_one_image(
            img_path, img_bgr, detector, pose_estimator, visualizer, args
        )
        combined_scores.append(payload)
        mmcv.imwrite(out_bgr, os.path.join(args.output_dir, name))

        if args.print_scores:
            if payload.get("error"):
                print(f"  scores: ERROR {payload['error']}")
            elif not payload.get("instances"):
                print(f"  scores: (no instances) {payload.get('note', '')}")
            else:
                for ins in payload["instances"]:
                    b = ins["bellguard"]
                    print(
                        f"  {ins['label']}: bellguard score={b['score']:.4f} "
                        f"xy=({b['x']:.1f},{b['y']:.1f})"
                    )

        if args.save_scores_json:
            stem, _ = os.path.splitext(name)
            score_path = os.path.join(args.output_dir, f"{stem}_scores.json")
            with open(score_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    if args.save_scores_json and combined_scores:
        all_path = os.path.join(args.output_dir, "all_scores.json")
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump(combined_scores, f, indent=2)
        print(f"Wrote combined scores: {all_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
