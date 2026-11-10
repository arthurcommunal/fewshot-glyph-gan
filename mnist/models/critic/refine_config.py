"""Write checkpoints/refine.json, the settings of the latent refinement done by generate.py.

generate.py moves each latent z by K steps of gradient ascent on the critic score before decoding it:
    z <- z + eps * grad_z C(G(z)) / k_eff.
k_eff is an estimate of the Lipschitz constant of z -> C(G(z)) (Tanaka, 2019, latent DOT):
    k_eff = max_{i != j} |C(G(z_i)) - C(G(z_j))| / ||z_i - z_j||   over n latents (seed 777).
Dividing by k_eff makes eps independent of the scale of the critic. It is computed once here, never in generate.py.

Final model (k_eff = 0.425477):
    python -m models.critic.refine_config --generator checkpoints/G.pth --critic checkpoints/critic.pth --eps 0.02 --K 20 --seed 201
"""
import argparse
import json

import torch

from mnist.critic_model import load_critic
from mnist.generate import load_generator, pick_device


@torch.no_grad()
def estimate_k_eff(generator, critic, device, n: int = 500, seed: int = 777) -> float:
    z = torch.randn(n, 100, generator=torch.Generator().manual_seed(seed)).to(device)
    scores = critic(generator(z)).flatten()
    distances = torch.cdist(z, z) + torch.eye(n, device=device)  # the diagonal gives 0 / 1 instead of 0 / 0
    return ((scores[:, None] - scores[None, :]).abs() / distances).max().item()


def main() -> None:
    parser = argparse.ArgumentParser("Estimate k_eff and write the refinement settings")
    parser.add_argument("--generator", type=str, default="checkpoints/G.pth")
    parser.add_argument("--critic", type=str, default="checkpoints/critic.pth")
    parser.add_argument("--eps", type=float, default=0.02, help="Step size")
    parser.add_argument("--K", type=int, default=20, help="Number of ascent steps")
    parser.add_argument("--seed", type=int, default=201, help="Seed of the latents drawn by generate.py")
    parser.add_argument("--n_pairs", type=int, default=500, help="Number of latents used to estimate k_eff")
    parser.add_argument("--out", type=str, default="checkpoints/refine.json")
    args = parser.parse_args()

    device = pick_device()
    generator = load_generator(args.generator, device)
    critic = load_critic(args.critic, device)
    k_eff = estimate_k_eff(generator, critic, device, n=args.n_pairs)

    spec = {"method": "ascent", "eps": args.eps, "K": args.K, "k_eff": round(k_eff, 6), "seed": args.seed}
    with open(args.out, "w") as f:
        json.dump(spec, f)
    print(f"{json.dumps(spec)} -> {args.out}")


if __name__ == "__main__":
    main()
