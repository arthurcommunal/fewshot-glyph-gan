"""Save a grid of generated digits, whatever the method used to train the generator.

Usage: python -m mnist.visualize runs/wgan_local/best_g.pth
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image

from mnist.model import Generator


@torch.no_grad()
def save_samples(generator: nn.Module, path: Path, n_samples: int = 10, z_dim: int = 100, seed: int = 0, scale: int = 4) -> None:
    """Save a grid of samples from any module mapping (B, z_dim) to images in [-1, 1].

    Args:
        generator: Module whose output can be reshaped to (B, 1, 28, 28).
        path: Where to save the PNG.
        n_samples: Number of samples in the grid.
        z_dim: Dimension of the latent space.
        seed: Seed of the latents, so that runs are compared on the same z.
        scale: Upscaling factor, 28x28 images being too small to read.
    """
    device = next(generator.parameters()).device
    # A local RNG leaves the global state untouched, even when called during training
    rng = torch.Generator().manual_seed(seed)
    z = torch.randn(n_samples, z_dim, generator=rng).to(device)

    generator.eval()
    images = generator(z).view(-1, 1, 28, 28)
    images = F.interpolate(images, scale_factor=scale, mode="nearest")
    save_image(images, path, nrow=5, normalize=True, value_range=(-1, 1), pad_value=1)


def load_generator(path: Path) -> nn.Module:
    """Load a Generator from a state_dict or from a full training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("generator", checkpoint)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    generator = Generator(g_output_dim=784)
    generator.load_state_dict(state_dict)
    return generator


def main() -> None:
    parser = argparse.ArgumentParser("Visualize samples of a trained generator")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Defaults to <checkpoint>_samples.png")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = args.out or args.checkpoint.with_name(f"{args.checkpoint.stem}_samples.png")
    save_samples(load_generator(args.checkpoint), out, args.n_samples, seed=args.seed)
    print(f"Saved {args.n_samples} samples to {out}")


if __name__ == "__main__":
    main()
