import copy
import torch
import torch.nn as nn
from .config import *

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

_DINO_BASE = None


def dino_backbone():
    """DINOv2 ViT-S/14 with the positional embedding fixed to the 224x224 grid (downloaded once via torch.hub)."""
    global _DINO_BASE
    if _DINO_BASE is None:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True,
                           skip_validation=True, verbose=False)
        grid = IMG_SIZE // m.patch_size
        with torch.no_grad():
            dummy = torch.zeros(1, 1 + grid * grid, m.embed_dim)
            pe = m.interpolate_pos_encoding(dummy, IMG_SIZE, IMG_SIZE).detach().clone()
        m.pos_embed = nn.Parameter(pe)
        _DINO_BASE = m.eval()
    return copy.deepcopy(_DINO_BASE)


class DinoClassifier(nn.Module):
    def __init__(self, n=len(CLASSES)):
        super().__init__()
        self.backbone = dino_backbone()
        self.head = nn.Linear(self.backbone.embed_dim, n)
    def embed(self, x):                      # normalised CLS token, 384-d
        return self.backbone(x)
    def forward(self, x): return self.head(self.embed(x))
