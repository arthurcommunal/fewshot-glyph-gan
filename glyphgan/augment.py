"""Hand-crafted baseline: "augmentation as a generator".

A sample = one random support image + random stroke-thickness change + random affine + elastic deformation
(Simard et al. 2003), all batched in torch (max-pool morphology, `grid_sample`), so it runs on GPU as well.

    sampler = AugmentSampler({"rot": 10})       # defaults overridden by the dict
    x = sampler(ep, n, g)                        # (n, 784) in [-1, 1]
    AugmentSampler(IDENTITY)(ep, n, g)           # returns support images unchanged

    python -m glyphgan.augment --out runs/augment/grid.png [--params runs/augment/best.json]

Parameters (all "strengths" are max amplitudes of symmetric uniform distributions; 0 disables the transform):
  rot      rotation, degrees            shear    shear angle (x direction), degrees
  scale    isotropic log-scale          aniso    anisotropic log-scale (sx = e^{+a}, sy = e^{-a})
  trans    translation, pixels          thick    stroke thickness: t ~ U(-thick, thick), blend towards
                                                 3x3 dilation (t > 0) or erosion (t < 0) with weight |t|
  alpha    elastic RMS displacement, pixels (the smoothed uniform field is normalised to unit RMS, so alpha is
           interpretable, unlike Simard's raw scale)
  sigma    elastic smoothing, Gaussian std in pixels
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

SIZE = 28
DEFAULTS = {"rot": 12.0, "shear": 12.0, "scale": 0.10, "aniso": 0.10, "trans": 1.5,
            "alpha": 0.8, "sigma": 3.0, "thick": 0.3}
IDENTITY = {k: 0.0 for k in DEFAULTS} | {"sigma": 3.0}
WEAK = IDENTITY | {"rot": 5.0, "scale": 0.05, "trans": 1.0}      # "only small affine"


def _uniform(shape, amp, g, device):
    return (torch.rand(shape, generator=g, device=g.device if g is not None else "cpu") * 2 - 1).to(device) * amp


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B, C, H, W) with replicate padding."""
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


class AugmentSampler:
    def __init__(self, params: dict | None = None, device=None):
        unknown = set(params or {}) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown augmentation params: {sorted(unknown)}")
        self.p = DEFAULTS | {k: float(v) for k, v in (params or {}).items()}
        self.device = device

    def __repr__(self):
        return f"AugmentSampler({self.p})"

    @torch.no_grad()
    def __call__(self, ep, n: int, g: torch.Generator | None = None) -> torch.Tensor:
        p, sup = self.p, ep.support
        dev = self.device or sup.device
        idx = torch.randint(len(sup), (n,), generator=g, device=g.device if g is not None else "cpu")
        u = (sup.to(dev)[idx.to(dev)].view(n, 1, SIZE, SIZE) + 1) / 2          # [0, 1], background = 0

        if p["thick"] > 0:
            t = _uniform((n, 1, 1, 1), p["thick"], g, dev)
            dil = F.max_pool2d(u, 3, 1, 1)
            ero = -F.max_pool2d(-u, 3, 1, 1)
            peak = u.amax(dim=(2, 3), keepdim=True)
            u = u + t.clamp(min=0) * (dil - u) + (-t).clamp(min=0) * (ero - u)
            # erosion of 1-2 px antialiased strokes mostly dims them: restore the original peak intensity
            u = u * (peak / u.amax(dim=(2, 3), keepdim=True).clamp(min=1e-3)).clamp(max=4.0)

        geo = [p[k] for k in ("rot", "shear", "scale", "aniso", "trans")]
        if any(v > 0 for v in geo) or p["alpha"] > 0:
            # theta maps output coords to input coords (normalised [-1, 1], align_corners=False).
            rot = torch.deg2rad(_uniform((n,), p["rot"], g, dev))
            sh = torch.tan(torch.deg2rad(_uniform((n,), p["shear"], g, dev)))
            s, a = _uniform((n,), p["scale"], g, dev), _uniform((n,), p["aniso"], g, dev)
            sx, sy = torch.exp(s + a), torch.exp(s - a)
            tr = _uniform((n, 2), p["trans"], g, dev) * 2 / SIZE
            cos, sin, z, o = torch.cos(rot), torch.sin(rot), torch.zeros_like(rot), torch.ones_like(rot)
            R = torch.stack([cos, -sin, sin, cos], 1).view(n, 2, 2)
            Sh = torch.stack([o, sh, z, o], 1).view(n, 2, 2)
            S = torch.stack([sx, z, z, sy], 1).view(n, 2, 2)
            theta = torch.cat([R @ Sh @ S, tr.unsqueeze(2)], 2)
            grid = F.affine_grid(theta, (n, 1, SIZE, SIZE), align_corners=False)
            if p["alpha"] > 0:
                field = _uniform((n, 2, SIZE, SIZE), 1.0, g, dev)
                field = _gaussian_blur(field, max(p["sigma"], 1e-3))
                field = field / field.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp(min=1e-8)
                grid = grid + field.permute(0, 2, 3, 1) * p["alpha"] * 2 / SIZE
            u = F.grid_sample(u, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        return (u.clamp(0, 1) * 2 - 1).view(n, -1)


def save_grid(params: dict | None, out: Path, n_chars: int = 6, rows: int = 3, root="data", seed: int = 0):
    """Per character: support row (10 images) followed by `rows` rows of 10 augmentations."""
    from torchvision.utils import save_image
    from glyphgan.data import episodes

    eps = episodes("dev", root=root, max_chars=n_chars, seed=0)
    g = torch.Generator().manual_seed(seed)
    sampler = AugmentSampler(params)
    blocks = []
    for ep in eps:
        blocks += [ep.support] + [sampler(ep, len(ep.support), g) for _ in range(rows)]
    x = torch.cat(blocks).view(-1, 1, SIZE, SIZE)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((x + 1) / 2, out, nrow=len(eps[0].support), padding=2, pad_value=0.3)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/augment/grid.png")
    ap.add_argument("--params", default=None, help="json file with params (or a {'params': ...} dict), or 'weak'/'identity'")
    ap.add_argument("--root", default="data")
    a = ap.parse_args()
    params = None
    if a.params in ("weak", "identity"):
        params = WEAK if a.params == "weak" else IDENTITY
    elif a.params:
        d = json.loads(Path(a.params).read_text())
        params = d.get("params", d)
    # sanity: identity must reproduce the support exactly
    from glyphgan.data import episodes
    ep = episodes("dev", root=a.root, max_chars=1)[0]
    g = torch.Generator().manual_seed(0)
    x = AugmentSampler(IDENTITY)(ep, 50, g)
    assert all((x[i:i + 1] == ep.support).all(1).any() for i in range(len(x))), "identity is not the identity"
    print("identity check ok;", AugmentSampler(params))
    print(save_grid(params, Path(a.out), root=a.root))
