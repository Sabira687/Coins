import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # CPU fallback for ops missing on Apple GPUs
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P1_DIR, P2_DIR = ROOT / "dataset", ROOT / "dataset_p2"

CLASSES = ["5_tenge", "10_tenge", "20_tenge", "50_tenge", "100_tenge"]
SHORT = ["5", "10", "20", "50", "100"]
C2I = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
