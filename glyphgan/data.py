"""Omniglot at 28x28, MNIST-like (white strokes on black, glyph cropped and centred in a 22x22 box, values in [-1, 1]).

    python -m glyphgan.data --root data      # downloads + builds data/omniglot28.pt once

Splits:
  - background (30 alphabets, 964 characters): pretraining of G, critic and feature classifier, tuning of the
    augmentation baseline.
  - evaluation (20 alphabets, 659 characters): few-shot episodes, never seen during pretraining. Its alphabets are
    split in two halves: "dev" (hyperparameter tuning) and "test" (final numbers only).
  - inside an episode, the 20 drawings of a character are split 10 "support" (adaptation) / 10 "heldout" (metrics).
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

SIZE = 28
CACHE = "omniglot28c.pt"  # c = cropped/centred


def _to_28(img: Image.Image, dilate: int, box: int = 22, min_side: int = 40) -> np.ndarray:
    """105x105 binary (black strokes) -> 28x28 uint8 with white strokes, MNIST-like: crop on the ink bounding box,
    fit it in a centred box x box square (MNIST: 20x20), then downsample with a box filter (area average =
    antialiased). Dilation keeps strokes visible after downsampling. min_side caps the zoom so that a tiny glyph
    (e.g. a dot) is not blown up to the full box."""
    x = Image.eval(img.convert("L"), lambda v: 255 - v)
    if dilate > 1:
        x = x.filter(ImageFilter.MaxFilter(dilate))
    a = np.asarray(x)
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return np.zeros((SIZE, SIZE), np.uint8)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1, min_side) + 2
    padded = np.pad(a, side)
    y0, x0 = int(round(cy - side / 2)) + side, int(round(cx - side / 2)) + side
    crop = Image.fromarray(padded[y0:y0 + side, x0:x0 + side])
    out = np.zeros((SIZE, SIZE), np.uint8)
    o = (SIZE - box) // 2
    out[o:o + box, o:o + box] = np.asarray(crop.resize((box, box), Image.BOX))
    return out


def build_cache(root: Path, dilate: int = 3) -> Path:
    from torchvision.datasets import Omniglot

    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, background in (("background", True), ("evaluation", False)):
        ds = Omniglot(root=str(root), background=background, download=True)
        alphabets = sorted({c.split("/")[0] for c in ds._characters})
        a_index = {a: i for i, a in enumerate(alphabets)}
        imgs, chars, alphas, drawers = [], [], [], []
        for (name, char_idx), (img, _) in zip(ds._flat_character_images, ds):
            imgs.append(_to_28(img, dilate))
            chars.append(char_idx)
            alphas.append(a_index[ds._characters[char_idx].split("/")[0]])
            drawers.append(int(name.split("_")[1].split(".")[0]))
        out[split] = {
            "images": torch.from_numpy(np.stack(imgs)),          # (N, 28, 28) uint8
            "char": torch.tensor(chars), "alphabet": torch.tensor(alphas), "drawer": torch.tensor(drawers),
            "char_names": list(ds._characters), "alphabet_names": alphabets,
        }
    out["dilate"] = dilate
    path = root / CACHE
    torch.save(out, path)
    return path


def _load(root: Path) -> dict:
    path = Path(root) / CACHE
    if not path.exists():
        build_cache(Path(root))
    return torch.load(path, weights_only=False)


def to_unit(images_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (N, 28, 28) -> float (N, 784) in [-1, 1], the tanh range of the generator."""
    return images_u8.float().div(255).mul(2).sub(1).reshape(len(images_u8), -1)


@dataclass
class Split:
    x: torch.Tensor          # (N, 784) in [-1, 1]
    char: torch.Tensor       # (N,) character index, local to the split
    alphabet: torch.Tensor   # (N,)
    char_names: list
    alphabet_names: list


def load_split(root="data", split="background") -> Split:
    d = _load(Path(root))[split]
    return Split(to_unit(d["images"]), d["char"], d["alphabet"], d["char_names"], d["alphabet_names"])


@dataclass
class Episode:
    char: int                # index in the evaluation split
    name: str                # "Alphabet/characterXX"
    alphabet: int
    support: torch.Tensor    # (10, 784) images used for adaptation
    heldout: torch.Tensor    # (10, 784) images used only for metrics


def eval_alphabets(part: str, root="data", seed: int = 0) -> list[int]:
    """Deterministic half split of the 20 evaluation alphabets into 'dev' and 'test'."""
    names = _load(Path(root))["evaluation"]["alphabet_names"]
    perm = np.random.default_rng(seed).permutation(len(names))
    half = len(names) // 2
    return sorted(perm[:half].tolist()) if part == "dev" else sorted(perm[half:].tolist())


def episodes(part: str = "dev", root="data", n_support: int = 10, seed: int = 0, max_chars: int | None = None) -> list[Episode]:
    """One episode per character of the chosen alphabets. The support/heldout split of the 20 drawings is a seeded
    permutation per character (same for every method, so all methods see exactly the same support images)."""
    ev = load_split(root, "evaluation")
    keep = set(eval_alphabets(part, root, seed=0))
    rng = np.random.default_rng(seed)
    out = []
    for c in range(len(ev.char_names)):
        idx = (ev.char == c).nonzero().squeeze(1)
        a = int(ev.alphabet[idx[0]])
        perm = idx[torch.from_numpy(rng.permutation(len(idx)))]   # always consume the rng -> stable across parts
        if a not in keep:
            continue
        out.append(Episode(c, ev.char_names[c], a, ev.x[perm[:n_support]], ev.x[perm[n_support:]]))
    if max_chars is not None:
        sel = np.random.default_rng(seed + 1).permutation(len(out))[:max_chars]
        out = [out[i] for i in sorted(sel)]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--dilate", type=int, default=3)
    a = p.parse_args()
    print(build_cache(Path(a.root), a.dilate))
