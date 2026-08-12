"""
ODTrack PyTorch engine — video-level temporal token tracking (AAAI 2024).

Requires one-time setup:
  python setup_odtrack.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

FENCING_DIR = Path(__file__).resolve().parent
ODTRACK_ROOT = FENCING_DIR / "vendor" / "odtrack"
MODELS_DIR = FENCING_DIR / "models" / "odtrack"
CHECKPOINT_NAME = "ODTrack_ep0300.pth.tar"
CONFIG_NAME = "baseline"
MIN_CHECKPOINT_BYTES = 300_000_000

_device: Optional[torch.device] = None
_odtrack_ready = False


def _valid_ckpt(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= MIN_CHECKPOINT_BYTES


def _write_local_py() -> None:
    local_py = ODTRACK_ROOT / "lib" / "test" / "evaluation" / "local.py"
    prj = str(ODTRACK_ROOT).replace("\\", "/")
    save = str(MODELS_DIR).replace("\\", "/")
    local_py.parent.mkdir(parents=True, exist_ok=True)
    local_py.write_text(
        "from lib.test.evaluation.environment import EnvSettings\n\n"
        "def local_env_settings():\n"
        "    settings = EnvSettings()\n"
        f"    settings.prj_dir = r'{prj}'\n"
        f"    settings.save_dir = r'{save}'\n"
        "    return settings\n",
        encoding="utf-8",
    )


def _ensure_odtrack_imports() -> None:
    global _odtrack_ready
    if _odtrack_ready:
        return
    if not ODTRACK_ROOT.is_dir():
        raise FileNotFoundError(
            f"ODTrack not installed at {ODTRACK_ROOT}. Run: python setup_odtrack.py"
        )
    root = str(ODTRACK_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _write_local_py()
    _odtrack_ready = True


def _pick_device() -> torch.device:
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = torch.device("cuda:0")
        print(f"ODTrack inference: CUDA ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        _device = torch.device("cpu")
        print("ODTrack inference: CPU (install CUDA PyTorch for GPU speed)", flush=True)
    return _device


def ensure_odtrack_model() -> str:
    dest = MODELS_DIR / CHECKPOINT_NAME
    if _valid_ckpt(dest):
        return str(dest)
    from setup_odtrack import download_checkpoint

    return str(download_checkpoint())


class _DevicePreprocessor:
    def __init__(self, device: torch.device):
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        from lib.utils.misc import NestedTensor

        img_tensor = torch.tensor(img_arr, device=self.device).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).to(self.device).unsqueeze(0)
        return NestedTensor(img_tensor_norm, amask_tensor)


def _load_processing_utils():
    path = ODTRACK_ROOT / "lib" / "train" / "data" / "processing_utils.py"
    spec = importlib.util.spec_from_file_location("odtrack_processing_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sample_target, mod.transform_image_to_crop


def _transform_bbox_to_crop(
    box_in,
    resize_factor,
    device: torch.device,
    template_size: int,
    transform_image_to_crop,
    box_extract=None,
):
    crop_sz = torch.Tensor([template_size, template_size])
    box_in_t = torch.tensor(box_in)
    box_extract_t = torch.tensor(box_extract if box_extract is not None else box_in)
    template_bbox = transform_image_to_crop(
        box_in_t, box_extract_t, resize_factor, crop_sz, normalize=True
    )
    return template_bbox.view(1, 1, 4).to(device)


class ODTrackEngine:
    """ODTrack inference with temporal memory — plain bbox output."""

    def __init__(self, checkpoint_path: Optional[str] = None, config_name: str = CONFIG_NAME):
        _ensure_odtrack_imports()
        self.device = _pick_device()
        self.provider = str(self.device)

        from lib.config.odtrack.config import cfg, update_config_from_file
        from lib.models.odtrack import build_odtrack
        from lib.test.utils.hann import hann2d
        from lib.utils.box_ops import clip_box
        from lib.utils.ce_utils import generate_mask_cond

        sample_target, transform_image_to_crop = _load_processing_utils()

        yaml_file = ODTRACK_ROOT / "experiments" / "odtrack" / f"{config_name}.yaml"
        update_config_from_file(str(yaml_file))
        self.cfg = cfg

        self.template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
        self.template_size = int(cfg.TEST.TEMPLATE_SIZE)
        self.search_factor = float(cfg.TEST.SEARCH_FACTOR)
        self.search_size = int(cfg.TEST.SEARCH_SIZE)

        ckpt = checkpoint_path or ensure_odtrack_model()
        self.network = build_odtrack(cfg, training=False)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.network.load_state_dict(state["net"], strict=True)
        self.network.to(self.device)
        self.network.eval()

        self.preprocessor = _DevicePreprocessor(self.device)
        self._sample_target = sample_target
        self._clip_box = clip_box
        self._generate_mask_cond = generate_mask_cond
        self._transform_image_to_crop = transform_image_to_crop

        feat_sz = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([feat_sz, feat_sz]).long(), centered=True).to(self.device)

        self._state: List[float] = [0.0, 0.0, 0.0, 0.0]
        self._frame_id = 0
        self.memory_frames: List[torch.Tensor] = []
        self.memory_masks: List[torch.Tensor] = []

    @staticmethod
    def _rgb(frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _tensor_on_device(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device) if t.device != self.device else t

    def initialize(self, image_bgr: np.ndarray, bbox_xywh: Tuple[float, float, float, float]) -> None:
        image = self._rgb(image_bgr)
        init_bbox = list(bbox_xywh)
        z_patch_arr, resize_factor, z_amask_arr = self._sample_target(
            image, init_bbox, self.template_factor, output_sz=self.template_size
        )
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        self.memory_frames = [template.tensors]
        self.memory_masks = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = _transform_bbox_to_crop(
                init_bbox,
                resize_factor,
                self.device,
                self.template_size,
                self._transform_image_to_crop,
            ).squeeze(1)
            self.memory_masks.append(
                self._generate_mask_cond(self.cfg, 1, self.device, template_bbox)
            )
        self._state = init_bbox
        self._frame_id = 0

    def _select_memory_frames(self) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        num_segments = int(self.cfg.TEST.TEMPLATE_NUMBER)
        cur_frame_idx = self._frame_id
        if num_segments != 1:
            dur = max(1, cur_frame_idx // num_segments)
            indexes = np.concatenate(
                [np.array([0]), np.array(list(range(num_segments))) * dur + dur // 2]
            )
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames: List[torch.Tensor] = []
        select_masks: List[torch.Tensor] = []
        for idx in indexes:
            select_frames.append(self._tensor_on_device(self.memory_frames[int(idx)]))
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                select_masks.append(self._tensor_on_device(self.memory_masks[int(idx)]))
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            return select_frames, torch.cat(select_masks, dim=1)
        return select_frames, None

    def _map_box_back(self, pred_box: List[float], resize_factor: float) -> List[float]:
        cx_prev = self._state[0] + 0.5 * self._state[2]
        cy_prev = self._state[1] + 0.5 * self._state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def track(self, image_bgr: np.ndarray) -> Tuple[List[float], float]:
        image = self._rgb(image_bgr)
        h, w = image.shape[:2]
        self._frame_id += 1

        x_patch_arr, resize_factor, x_amask_arr = self._sample_target(
            image, self._state, self.search_factor, output_sz=self.search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        box_mask_z: Optional[torch.Tensor] = None
        if self._frame_id <= int(self.cfg.TEST.TEMPLATE_NUMBER):
            template_list = [self._tensor_on_device(f) for f in self.memory_frames]
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = torch.cat([self._tensor_on_device(m) for m in self.memory_masks], dim=1)
        else:
            template_list, box_mask_z = self._select_memory_frames()

        with torch.no_grad():
            out_dict = self.network.forward(
                template=template_list, search=[search.tensors], ce_template_mask=box_mask_z
            )
        if isinstance(out_dict, list):
            out_dict = out_dict[-1]

        pred_score_map = out_dict["score_map"]
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict["size_map"], out_dict["offset_map"])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()
        self._state = self._clip_box(self._map_box_back(pred_box, resize_factor), h, w, margin=10)
        score = float(response.max().detach().cpu().item())

        z_patch_arr, z_resize_factor, z_amask_arr = self._sample_target(
            image, self._state, self.template_factor, output_sz=self.template_size
        )
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors
        if self._frame_id > int(self.cfg.TEST.MEMORY_THRESHOLD):
            frame = frame.detach().cpu()
        self.memory_frames.append(frame)
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            dev = frame.device if frame.device.type != "cpu" else self.device
            template_bbox = _transform_bbox_to_crop(
                self._state,
                z_resize_factor,
                dev,
                self.template_size,
                self._transform_image_to_crop,
            ).squeeze(1)
            mask = self._generate_mask_cond(self.cfg, 1, dev, template_bbox)
            if self._frame_id > int(self.cfg.TEST.MEMORY_THRESHOLD):
                mask = mask.detach().cpu()
            self.memory_masks.append(mask)

        return self._state, score

    def set_state(self, bbox_xywh: Tuple[float, float, float, float]) -> None:
        self._state = list(bbox_xywh)


if __name__ == "__main__":
    path = ensure_odtrack_model()
    print(f"ODTrack ready: {path}")
