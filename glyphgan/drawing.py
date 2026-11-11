"""Turn a free-hand canvas drawing into the 28x28 format of the training data.

Omniglot glyphs are 105x105 renderings of online strokes: about 65 px wide, strokes about 5 px thick. A canvas
drawing has an arbitrary position, size and brush width. To land in the same distribution we
  1. crop the ink and rescale it so that its longer side is OMNI_SIDE px (the median Omniglot glyph size),
  2. thin the strokes to a 1-px skeleton and re-thicken them with a disk of radius RADIUS (calibrated so that real
     Omniglot glyphs survive this round trip almost unchanged, see calibrate()),
  3. apply exactly the training preprocessing, glyphgan.data._to_px (crop, centre in a 22x22 box, dilate, BOX).
Position and margins do not matter: _to_px crops on the ink bounding box anyway.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

from glyphgan.data import _to_px

OMNI_SIDE = 65
RADIUS = 2
CANVAS = 105


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def ink_mask(canvas: np.ndarray) -> np.ndarray:
    """Canvas array (H, W), (H, W, 3) or (H, W, 4) with dark strokes on a light or transparent background -> bool."""
    a = np.asarray(canvas)
    if a.ndim == 2:
        return a < 128
    if a.shape[2] == 4:
        alpha = a[..., 3]
        if alpha.min() < 255:                       # transparent background: ink = painted pixels
            return (alpha > 127) & (a[..., :3].mean(-1) < 128)
    return a[..., :3].mean(-1) < 128


def normalize_strokes(mask: np.ndarray, side: int = OMNI_SIDE, radius: int = RADIUS) -> Image.Image:
    """Bool ink mask -> 105x105 'L' image, black strokes on white, Omniglot-like glyph size and stroke width."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return Image.new("L", (CANVAS, CANVAS), 255)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = side / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = np.asarray(Image.fromarray(crop.astype(np.uint8) * 255).resize((nw, nh), Image.BILINEAR)) > 64
    skel = skeletonize(small) if small.sum() > 1 else small
    out = np.zeros((CANVAS, CANVAS), bool)
    y0, x0 = (CANVAS - nh) // 2, (CANVAS - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = skel
    out = binary_dilation(out, _disk(radius))
    return Image.fromarray(np.where(out, 0, 255).astype(np.uint8))


def canvas_to_px(canvas: np.ndarray, dilate: int = 3, size: int = 28) -> np.ndarray:
    """Canvas drawing -> (size, size) uint8, white strokes on black, exactly like the cached training data."""
    return _to_px(normalize_strokes(ink_mask(canvas)), dilate, size)


def calibrate(paths: list[str], radii=(1, 2, 3)) -> dict:
    """Mean absolute pixel error (on the 0..1 scale) between _to_px(original) and _to_px(round trip) for real Omniglot
    PNGs, for each candidate radius. The round trip deliberately ignores the original size (rescaled to OMNI_SIDE)."""
    res = {}
    for r in radii:
        errs = []
        for p in paths:
            img = Image.open(p)
            ref = _to_px(img, 3).astype(float) / 255
            rt = _to_px(normalize_strokes(np.asarray(img.convert("L")) < 128, radius=r), 3).astype(float) / 255
            errs.append(np.abs(ref - rt).mean())
        res[r] = float(np.mean(errs))
    return res
