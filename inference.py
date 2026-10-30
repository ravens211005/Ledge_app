"""Model loading and inference for YOLO segmentation + Depth Anything V2 Small (CPU only)."""
import config  # noqa: F401  (must be imported before torch: hides GPUs)

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from ultralytics import YOLO

from depth_anything_v2.dpt import DepthAnythingV2

log = logging.getLogger("inference")
torch.set_num_threads(config.NUM_THREADS)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InputError(Exception):
    """Raised for invalid user input; the message is safe to show in the UI."""


@dataclass
class Detection:
    idx: int
    class_id: int
    class_name: str
    conf: float
    box: tuple          # (x1, y1, x2, y2) in working-image pixels
    mask: np.ndarray    # bool HxW, same size as the working image


@dataclass
class InferenceResult:
    rgb: np.ndarray             # working-resolution RGB image
    detections: list
    depth_raw: np.ndarray       # HxW float32, relative inverse depth (larger = closer), NOT metres
    depth_rel: np.ndarray       # HxW float32, per-image normalised to 0..1 (robust percentiles)
    timings: dict


class Models:
    def __init__(self):
        for p in (config.YOLO_WEIGHTS, config.DEPTH_CHECKPOINT):
            if not p.is_file():
                raise FileNotFoundError(f"Model file not found: {p}")

        log.info("Loading YOLO weights: %s", config.YOLO_WEIGHTS)
        self.yolo = YOLO(str(config.YOLO_WEIGHTS))
        self.names = dict(self.yolo.names)
        log.info("YOLO task=%s classes=%s", self.yolo.task, self.names)
        if self.yolo.task != "segment":
            raise RuntimeError(f"Expected a segmentation model, got task='{self.yolo.task}'")
        missing = {config.STAIR_CLASS_NAME, config.EDGE_CLASS_NAME} - set(self.names.values())
        if missing:
            raise RuntimeError(f"Class name(s) {missing} not in model.names={self.names}. "
                               "Fix STAIR_CLASS_NAME / EDGE_CLASS_NAME in config.py.")

        log.info("Loading Depth Anything V2 Small: %s", config.DEPTH_CHECKPOINT)
        self.depth = DepthAnythingV2(**config.DEPTH_ENCODER_CFG)
        state = torch.load(str(config.DEPTH_CHECKPOINT), map_location="cpu", weights_only=True)
        self.depth.load_state_dict(state)
        self.depth.eval()  # stays on CPU

        self.lock = threading.Lock()  # one inference at a time; queue handles the rest
        self._warmup()

    def _warmup(self):
        log.info("Warming up models...")
        dummy = np.random.randint(0, 255, (480, 360, 3), dtype=np.uint8)
        self.run(dummy)
        log.info("Warm-up done.")

    # ------------------------------------------------------------------ YOLO
    def _run_yolo(self, rgb, conf_thres):
        h, w = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # Ultralytics expects BGR for numpy input
        with torch.inference_mode():
            r = self.yolo.predict(bgr, imgsz=config.YOLO_IMGSZ, conf=conf_thres,
                                  iou=config.YOLO_IOU, device="cpu", verbose=False)[0]
        dets = []
        if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
            return dets
        for i, (cls, score, box) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                                   r.boxes.xyxy.tolist())):
            poly = r.masks.xy[i]  # polygon in original (working) image coordinates
            if poly is None or len(poly) < 3:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
            dets.append(Detection(i, int(cls), self.names[int(cls)], float(score),
                                  tuple(box), m.astype(bool)))
        return dets

    # ----------------------------------------------------------------- Depth
    def _run_depth(self, rgb):
        """Own preprocessing so the tensor is guaranteed to stay on CPU (the repo's infer_image
        auto-selects CUDA, which would mismatch a CPU model on a machine with a GPU)."""
        h, w = rgb.shape[:2]
        s = config.DEPTH_INPUT_SIZE / max(h, w)
        nh = max(14, int(round(h * s / 14)) * 14)
        nw = max(14, int(round(w * s / 14)) * 14)
        x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
        with torch.inference_mode():
            d = self.depth(t)                                   # (1, nh, nw)
            d = F.interpolate(d[:, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        return d.numpy().astype(np.float32)

    @staticmethod
    def _normalise(d):
        lo, hi = np.percentile(d, [1, 99])
        if hi - lo < 1e-6:
            return np.zeros_like(d)
        return np.clip((d - lo) / (hi - lo), 0, 1).astype(np.float32)

    # ---------------------------------------------------------------- Public
    def run(self, rgb, conf=None):
        conf_thres = config.YOLO_CONF if conf is None else float(conf)
        with self.lock:
            t0 = time.perf_counter()
            dets = self._run_yolo(rgb, conf_thres)
            t1 = time.perf_counter()
            depth_raw = self._run_depth(rgb)
            t2 = time.perf_counter()
            depth_rel = self._normalise(depth_raw)
            t3 = time.perf_counter()
        return InferenceResult(rgb, dets, depth_raw, depth_rel,
                               {"yolo_s": t1 - t0, "depth_s": t2 - t1,
                                "normalise_s": t3 - t2, "total_s": t3 - t0})


_models = None
_load_lock = threading.Lock()


def load_models():
    global _models
    with _load_lock:
        if _models is None:
            _models = Models()
    return _models


def prepare_image(pil_img):
    """Validate and convert an uploaded PIL image to a working-resolution RGB uint8 array."""
    if pil_img is None:
        raise InputError("No image uploaded.")
    fmt = getattr(pil_img, "format", None)
    if fmt is not None and fmt.upper() not in config.ALLOWED_FORMATS:
        raise InputError(f"Unsupported format '{fmt}'. Use JPG, JPEG or PNG.")
    try:
        img = ImageOps.exif_transpose(pil_img).convert("RGB")
        w, h = img.size
        if w * h > config.MAX_PIXELS:
            raise InputError(f"Image too large ({w}x{h}).")
        if min(w, h) < config.MIN_SIDE:
            raise InputError(f"Image too small ({w}x{h}); minimum side is {config.MIN_SIDE}px.")
        rgb = np.asarray(img)
    except InputError:
        raise
    except Exception as e:
        raise InputError(f"Could not read the image (corrupted or unsupported): {e}")

    scale = config.MAX_SIDE / max(h, w)
    if scale < 1:  # downscale only, aspect ratio preserved
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)