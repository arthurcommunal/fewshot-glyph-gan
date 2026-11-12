"""Train a critic post hoc, on a frozen generator, for the latent refinement of generate.py.

The critic is the CNN of critic_model.py, trained with the WGAN-GP loss (Gulrajani et al., 2017):
    L = E[C(G(z))] - E[C(x)] + lambda * E[(||grad C(x_hat)|| - 1)^2],   x_hat = a x + (1 - a) G(z).
The generator is never updated. Final model:
    python -m mnist.models.critic.train_critic --generator runs/T2_clip_cont/best_ema_g.pth \
        --init_from runs/critic_5e5/critic_18000.pth --n_steps 100000 --run_name critic_t2
then copy runs/critic_t2/critic_98000.pth to checkpoints/critic.pth. The checkpoint was chosen among 30k, 60k and 98k
steps on validation Inception FID (metrics/fid_inception.py). Our critic started from a critic trained 18k steps on the
5e5 generator, and was trained on the EMA generator of T2 taken mid-run (best EMA at that time); it is used unchanged
with T4, the final generator.
Writes runs/<run_name>/{config.json, history.json, critic_<step>.pth}.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from mnist.critic_model import CNNCritic
from mnist.model import Generator
from mnist.models.wgan.wgan import IndexSampler, subset_to_tensor
from mnist.utils import get_mnist_splits


def load_generator_weights(path: str, device: torch.device) -> Generator:
    """Frozen generator from a plain state_dict (best_g.pth, final_ema_g.pth) or a last.pth (EMA weights if present)."""
    state = torch.load(path, map_location="cpu")
    for key in ("generator_ema", "generator"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    generator = Generator(g_output_dim=784)
    generator.load_state_dict(state)
    generator.to(device).eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    return generator


def gradient_penalty(critic: CNNCritic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, device=real.device)
    x_hat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    grad = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def em_estimate(critic: CNNCritic, real: torch.Tensor, fake: torch.Tensor, batch_size: int = 2000) -> float:
    """E[C(x)] - E[C(G(z))], the critic's estimate of the Wasserstein-1 distance."""
    score = lambda x: torch.cat([critic(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]).mean()
    return (score(real) - score(fake)).item()


def main() -> None:
    parser = argparse.ArgumentParser("Train a CNN critic with gradient penalty on a frozen generator")
    parser.add_argument("--generator", type=str, required=True, help="Generator weights (kept frozen)")
    parser.add_argument("--init_from", type=str, default=None, help="Optional critic to start from")
    parser.add_argument("--data_path", type=str, default="data/mnist")
    parser.add_argument("--n_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--gp_lambda", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=64, help="Channels of the first convolution")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    out_dir = Path("runs").joinpath(args.run_name)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists")
    out_dir.mkdir(parents=True)
    with open(out_dir.joinpath("config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train, val = get_mnist_splits(Path(args.data_path), download=not Path(args.data_path).exists(), seed=42, n_val=10000)
    x_train, x_val = subset_to_tensor(train, device), subset_to_tensor(val, device)

    generator = load_generator_weights(args.generator, device)
    critic = CNNCritic(w=args.width).to(device)
    if args.init_from:
        state = torch.load(args.init_from, map_location="cpu")
        critic.load_state_dict(state.get("state_dict", state))
    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # fixed fake images to monitor the Wasserstein estimate on train and validation
    with torch.no_grad():
        fake_train = generator(torch.randn(10000, 100, generator=torch.Generator().manual_seed(11)).to(device))
        fake_val = generator(torch.randn(10000, 100, generator=torch.Generator().manual_seed(12)).to(device))

    sampler = IndexSampler(x_train.size(0), device)
    history, start = [], time.time()
    critic.train()
    for step in range(1, args.n_steps + 1):
        real = x_train[sampler.sample(args.batch_size)]
        with torch.no_grad():
            fake = generator(torch.randn(args.batch_size, 100, device=device))
        scores = critic(torch.cat([real, fake]))
        loss = scores[args.batch_size:].mean() - scores[:args.batch_size].mean()
        loss = loss + args.gp_lambda * gradient_penalty(critic, real, fake)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.save_every == 0 or step == args.n_steps:
            critic.eval()
            row = {"step": step,
                   "train_em": em_estimate(critic, x_train[:10000], fake_train),
                   "val_em": em_estimate(critic, x_val, fake_val),
                   "seconds": time.time() - start}
            critic.train()
            history.append(row)
            print(json.dumps(row))
            torch.save({"kind": "cnn", "state_dict": critic.state_dict()}, out_dir.joinpath(f"critic_{step}.pth"))
            with open(out_dir.joinpath("history.json"), "w") as f:
                json.dump(history, f, indent=2)
            start = time.time()


if __name__ == "__main__":
    main()
