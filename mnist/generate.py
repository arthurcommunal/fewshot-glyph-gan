"""Generate 10 000 MNIST samples as PNGs in samples/ (0.png ... 9999.png).

    python generate.py --batch_size 256

Loads checkpoints/G.pth (the fixed Generator). If checkpoints/critic.pth AND checkpoints/refine.json exist,
each latent z ~ N(0, I) is first refined by gradient ascent on the critic score (the generator is unchanged):
    z <- z + eps * grad_z C(G(z)) / k_eff,   K steps,
with eps, K, k_eff and the latent seed read from refine.json (k_eff is fixed there, never re-estimated).
Otherwise samples come from G alone. PNGs store (x + 1) / 2, i.e. the tanh range [-1, 1] mapped to [0, 255]."""
import argparse
import json
import os
import time

import torch
import torchvision

from mnist.model import Generator

N_SAMPLES = 10000
DEFAULT_SEED = 101
CKPT_DIR = "checkpoints"


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_generator(path, device):
    sd = torch.load(path, map_location="cpu")
    for key in ("G_ema999", "generator", "G"):
        if key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    G = Generator(g_output_dim=784)
    G.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    G.to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G


def load_refinement(device):
    """Returns (critic, spec) or (None, None) when checkpoints/ holds no critic + refine.json."""
    c_path, j_path = os.path.join(CKPT_DIR, "critic.pth"), os.path.join(CKPT_DIR, "refine.json")
    if not (os.path.exists(c_path) and os.path.exists(j_path)):
        if os.path.exists(c_path) or os.path.exists(j_path):
            print("WARNING: critic.pth and refine.json must both be present; generating without refinement.")
        return None, None
    from critic_model import load_critic
    with open(j_path) as f:
        spec = json.load(f)
    if spec.get("method", "ascent") not in ("ascent", "dot", "none"):
        raise ValueError(f"unknown refinement method {spec['method']!r}")
    if spec.get("method") == "none":
        return None, spec
    for key in ("eps", "K", "k_eff"):
        if key not in spec:
            raise KeyError(f"refine.json needs '{key}'")
    return load_critic(c_path, device), spec


def refine(G, C, z, spec):
    """Latent refinement on one batch. ascent: z += eps*grad C(G(z))/k_eff.
    dot (Tanaka 2019): z -= eps*grad(||z - z0|| - C(G(z))/k_eff), from z = z0."""
    eps, K, k = float(spec["eps"]), int(spec["K"]), float(spec["k_eff"])
    z0 = z
    with torch.enable_grad():
        for _ in range(K):
            z = z.detach().requires_grad_(True)
            score = C(G(z)).sum() / k
            if spec.get("method", "ascent") == "dot":
                obj = ((z - z0).pow(2).sum(1) + 1e-12).sqrt().sum() - score
                z = z - eps * torch.autograd.grad(obj, z)[0]
            else:
                z = z + eps * torch.autograd.grad(score, z)[0]
    return z.detach()


def to_png_range(x, mapping):
    """faithful: [-1, 1] -> [0, 1] linearly. starter: the original script's behaviour (values < 0 clipped to 0)."""
    return (x + 1) / 2 if mapping == "faithful" else x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 10000 MNIST samples.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Number of samples per batch.")
    parser.add_argument("--png", choices=["faithful", "starter"], default="faithful", help="Pixel mapping of the PNGs.")
    parser.add_argument("--device", default=None, help="Force a device (default: cuda, then mps, then cpu).")
    args = parser.parse_args()

    t_start = time.time()
    device = pick_device(args.device)
    print(f"Using device: {device}")

    print("Model Loading...")
    G = load_generator(os.path.join(CKPT_DIR, "G.pth"), device)
    C, spec = load_refinement(device)
    seed = int(spec.get("seed", DEFAULT_SEED)) if spec else DEFAULT_SEED
    if C is not None:
        print(f"Latent refinement: {spec.get('method', 'ascent')} eps={spec['eps']} K={spec['K']} k_eff={spec['k_eff']}")
    else:
        print("No refinement: samples from the generator alone.")
    print(f"Model loaded. Latent seed {seed}, PNG mapping {args.png}.")

    # All latents drawn at once on the CPU: the same z on every device and for every batch size.
    z_all = torch.randn(N_SAMPLES, 100, generator=torch.Generator().manual_seed(seed))

    print("Start Generating")
    os.makedirs("samples", exist_ok=True)
    t_compute, t_write = 0.0, 0.0
    n_samples = 0
    for i in range(0, N_SAMPLES, args.batch_size):
        t0 = time.time()
        z = z_all[i:i + args.batch_size].to(device)
        if C is not None:
            z = refine(G, C, z, spec)
        with torch.no_grad():
            x = to_png_range(G(z), args.png).reshape(-1, 1, 28, 28).cpu()
        sync(device)
        t1 = time.time()
        for k in range(x.shape[0]):
            torchvision.utils.save_image(x[k], os.path.join("samples", f"{n_samples}.png"))
            n_samples += 1
        t_compute += t1 - t0
        t_write += time.time() - t1

    print(f"Wrote {n_samples} images to samples/.")
    print(f"Timing: total {time.time() - t_start:.1f} s (generation{' + refinement' if C is not None else ''} "
          f"{t_compute:.1f} s, PNG writing {t_write:.1f} s) on {device}.")
