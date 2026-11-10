"""Feature extractor for the metrics: small CNN classifier trained on the 964 background characters.

Embedding = penultimate layer (128-d), L2-normalized. The head is a cosine classifier, so the training objective acts
directly on the normalized embedding that the metrics use (Euclidean distance on the unit sphere).
"""
from pathlib import Path

import math
import torch
import torch.nn.functional as F
from torch import nn

DIM = 128


def _block(cin: int, cout: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class FeatureNet(nn.Module):
    """28x28 -> 128-d unit embedding; `logits` = scaled cosine similarity to the class weights."""

    def __init__(self, n_classes: int = 964, dim: int = DIM, scale: float = 16.0):
        super().__init__()
        self.body = nn.Sequential(
            *_block(1, 32), *_block(32, 32), nn.MaxPool2d(2),        # 14x14
            *_block(32, 64), *_block(64, 64), nn.MaxPool2d(2),       # 7x7
            *_block(64, 128), nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, dim),
        )
        self.weight = nn.Parameter(torch.randn(n_classes, dim) / math.sqrt(dim))
        self.scale = scale

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 784) or (N, 1, 28, 28) in [-1, 1] -> (N, dim) L2-normalized (differentiable)."""
        return F.normalize(self.body(x.reshape(-1, 1, 28, 28)), dim=1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.embed(x) @ F.normalize(self.weight, dim=1).T


def augment(x: torch.Tensor, g: torch.Generator | None = None) -> torch.Tensor:
    """Light random affine on a batch (N, 784) in [-1, 1]: rotation +-10 deg, scale 0.9-1.1, shift +-2 px."""
    n, dev = len(x), x.device
    u = lambda lo, hi: torch.rand(n, generator=g, device=dev) * (hi - lo) + lo  # noqa: E731
    a, s = u(-10, 10) * math.pi / 180, u(0.9, 1.1)
    tx, ty = u(-2, 2) * 2 / 28, u(-2, 2) * 2 / 28
    theta = torch.stack([torch.stack([a.cos() / s, -a.sin() / s, tx], 1),
                         torch.stack([a.sin() / s, a.cos() / s, ty], 1)], 1)
    img = x.reshape(n, 1, 28, 28) + 1                                  # background 0 so zero padding is black
    grid = F.affine_grid(theta, list(img.shape), align_corners=False)
    return (F.grid_sample(img, grid, align_corners=False) - 1).reshape(n, -1)


class FeatureExtractor:
    """Frozen embedding: `extractor(x)` with x (N, 784) in [-1, 1] -> (N, 128) unit-norm tensor, no grad."""

    def __init__(self, net: FeatureNet, device: str | torch.device = "cpu"):
        self.net = net.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.dim = net.weight.shape[1]

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "FeatureExtractor":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = FeatureNet(ck["n_classes"], ck["dim"])
        net.load_state_dict(ck["state_dict"])
        return cls(net, device)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, batch: int = 4096) -> torch.Tensor:
        out = [self.net.embed(x[i:i + batch].to(self.device, torch.float32)) for i in range(0, len(x), batch)]
        return torch.cat(out) if out else torch.empty(0, self.dim, device=self.device)
