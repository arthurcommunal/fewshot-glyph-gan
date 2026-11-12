"""Inception FID of the PNGs written by generate.py, our main metric for the final model.

Standard FID (Heusel et al., 2017): 2048-d pool features of Inception-v3 from pytorch-fid (pip install pytorch-fid;
its weights are downloaded on first use). Grey 28x28 images are repeated on 3 channels and resized to 299x299 by
the network. The real images are read from the MNIST files as uint8, exactly like the generated PNGs.

    python -m mnist.generate --batch_size 256
    python -m mnist.metrics.fid_inception --samples_dir samples --ref train10k   # tuning (first 10k training images)
    python -m mnist.metrics.fid_inception --samples_dir samples --ref test        # final check only
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from mnist.metrics.metrics import fid
from mnist.utils import get_mnist_splits


@torch.no_grad()
def inception_features(images: torch.Tensor, device: torch.device, batch_size: int = 100) -> np.ndarray:
    """images: (N, 28, 28) uint8 -> (N, 2048) Inception-v3 pool features."""
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as e:
        raise ImportError("Inception FID needs pytorch-fid: pip install pytorch-fid") from e
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].float().div(255).view(-1, 1, 28, 28).repeat(1, 3, 1, 1).to(device)
        features.append(model(batch)[0].flatten(1).cpu().double())
    return torch.cat(features).numpy()


def load_pngs(samples_dir: Path) -> torch.Tensor:
    files = sorted(samples_dir.glob("*.png"), key=lambda f: int(f.stem) if f.stem.isdigit() else f.stem)
    if not files:
        raise FileNotFoundError(f"No PNG in {samples_dir}")
    return torch.from_numpy(np.stack([np.array(Image.open(f).convert("L")) for f in files]))


def load_reference(ref: str, data_path: Path) -> torch.Tensor:
    if ref == "test":
        return datasets.MNIST(root=data_path, train=False, download=False).data
    if ref == "train10k":
        return datasets.MNIST(root=data_path, train=True, download=False).data[:10000]
    if ref == "val":
        _, val = get_mnist_splits(data_path, download=False, seed=42, n_val=10000)
        return val.dataset.data[val.indices]
    raise ValueError(f"unknown reference {ref!r}")


def main() -> None:
    parser = argparse.ArgumentParser("Inception FID between a folder of generated PNGs and real MNIST images")
    parser.add_argument("--samples_dir", type=str, default="samples")
    parser.add_argument("--ref", choices=["train10k", "val", "test"], default="train10k")
    parser.add_argument("--data_path", type=str, default="data/mnist")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    fake = load_pngs(Path(args.samples_dir))
    real = load_reference(args.ref, Path(args.data_path))
    score = fid(inception_features(real, device), inception_features(fake, device))
    print(f"Inception FID ({len(fake)} samples vs {args.ref}, {len(real)} images): {score:.3f}")


if __name__ == "__main__":
    main()
