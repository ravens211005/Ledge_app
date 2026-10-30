"""All tunable settings live here. Override any of them with environment variables."""
import os
from pathlib import Path

# Hide GPUs so local runs behave like Cloud Run (CPU-only). Must happen before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
BASE_DIR = Path(__file__).resolve().parent

# --- Model locations (configurable; later these can point at a Cloud Storage mount) ---
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
YOLO_WEIGHTS = Path(os.environ.get("YOLO_WEIGHTS", MODEL_DIR / "best.pt"))
DEPTH_CHECKPOINT = Path(os.environ.get("DEPTH_CHECKPOINT", MODEL_DIR / "depth_anything_v2_vits.pth"))

# --- Class names are looked up by NAME in model.names, never by hardcoded ID ---
STAIR_CLASS_NAME = os.environ.get("STAIR_CLASS_NAME", "stairs")
EDGE_CLASS_NAME = os.environ.get("EDGE_CLASS_NAME", "edges")

# --- YOLO ---
YOLO_CONF = float(os.environ.get("YOLO_CONF", 0.35))
YOLO_IOU = float(os.environ.get("YOLO_IOU", 0.7))
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", 640))   # use the size you trained at

# --- Depth Anything V2 Small ---
DEPTH_INPUT_SIZE = int(os.environ.get("DEPTH_INPUT_SIZE", 518))  # longest side fed to the model (multiple of 14 enforced)
DEPTH_ENCODER_CFG = dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384])

# --- Input validation / preprocessing ---
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 40_000_000))  # reject anything larger
MIN_SIDE = int(os.environ.get("MIN_SIDE", 64))
MAX_SIDE = int(os.environ.get("MAX_SIDE", 1280))            # larger images are downscaled (aspect preserved)
ALLOWED_FORMATS = {"JPEG", "PNG"}

# --- Edge depth sampling (same idea as your EDGE_DEPTH_BOX_BAND_FRACTION) ---
EDGE_DEPTH_BOX_BAND_FRACTION = float(os.environ.get("EDGE_DEPTH_BOX_BAND_FRACTION", 0.3))

# --- CPU / concurrency ---
NUM_THREADS = int(os.environ.get("NUM_THREADS", max(1, (os.cpu_count() or 2))))
QUEUE_MAX_SIZE = int(os.environ.get("QUEUE_MAX_SIZE", 8))

# --- Server (Cloud Run supplies PORT) ---
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 7860))