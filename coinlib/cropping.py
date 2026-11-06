"""Coin detection and cropping: the same code that built the datasets (extract_frames_p2.py)."""
import sys
from .config import ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extract_frames_p2 import locate_coin, crop_coin   # noqa: E402,F401
