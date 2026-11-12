"""Continue a WGAN training from a checkpoint (warm restart) and keep an EMA of the generator.

Used for the runs T2 and T4 of the final model:
    T2: python -m mnist.models.wgan.train_wgan_restart --init_from runs/wgan_local_step5e5/last.pth --lr 2e-5 --n_steps 600000 --run_name T2_clip_cont
    T4: python -m mnist.models.wgan.train_wgan_restart --init_from runs/T2_clip_cont/last.pth --lr 1e-5 --n_steps 548000 --run_name T4_clip_cont2

The generator, the critic and the RMSProp states are loaded from --init_from (a last.pth written by train_wgan.py
or by this script); only the learning rate restarts: linear warmup, then cosine decay to 1 % of --lr.
The EMA of the generator weights (decay --ema) is evaluated next to the raw weights and is the model we keep.
Writes runs/<run_name>/{config.json, history.json, last.pth, best_ema_g.pth, final_ema_g.pth}.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from mnist.metrics.classifier import MNISTClassifier
from mnist.metrics.metrics import extract_features
from mnist.model import Generator
from mnist.models.wgan.wgan import (
    Critic,
    IndexSampler,
    clip_weights,
    critic_loss,
    evaluate_model,
    generator_loss,
    subset_to_tensor,
)
from mnist.utils import get_mnist_splits


def lr_factor(step: int, n_steps: int, warmup_steps: int, min_factor: float) -> float:
    """Linear warmup from 1 %, then cosine decay from 1 to min_factor at the last step."""
    if step < warmup_steps:
        return 0.01 + 0.99 * step / warmup_steps
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser("Continue a WGAN training (warm restart) with an EMA of the generator")
    parser.add_argument("--init_from", type=str, required=True, help="last.pth of a previous run")
    parser.add_argument("--data_path", type=str, default="data/mnist")
    parser.add_argument("--n_steps", type=int, required=True, help="Number of generator updates")
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--critic_clip", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate after the warmup (generator and critic)")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--min_factor", type=float, default=0.01, help="Final learning rate / lr")
    parser.add_argument("--ema", type=float, default=0.999, help="EMA decay of the generator weights")
    parser.add_argument("--k", type=int, default=3, help="For FID, precision and recall estimators")
    parser.add_argument("--eval_every", type=int, default=25000)
    parser.add_argument("--n_eval", type=int, default=10000)
    parser.add_argument("--eval_seed", type=int, default=1234, help="Fixed latents for the evaluations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    if args.n_eval > 10000:
        raise ValueError("n_eval must be lower or equal to 10000")

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

    # data and classifier (same protocol as train_wgan.py, but fixed evaluation latents)
    train, val = get_mnist_splits(Path(args.data_path), download=not Path(args.data_path).exists(), seed=42, n_val=10000)
    x_train = subset_to_tensor(train, device)
    x_val = subset_to_tensor(val, device)[:args.n_eval]
    clf = MNISTClassifier()
    clf.load_state_dict(torch.load("runs/clf_local/best.pth", map_location=device))
    clf.to(device).eval()
    feat_real = extract_features(x_val, clf, batch_size=128, device=device)
    z_eval = torch.randn(args.n_eval, 100, generator=torch.Generator().manual_seed(args.eval_seed)).to(device)

    # modules and optimizers, restored from the previous run
    generator = Generator(g_output_dim=784).to(device)
    critic = Critic(input_dim=784).to(device)
    generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=args.lr)
    state = torch.load(args.init_from, map_location=device)
    generator.load_state_dict(state["generator"])
    critic.load_state_dict(state["critic"])
    generator_optimizer.load_state_dict(state["generator_optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])

    # the EMA starts from the loaded (raw) generator weights
    generator_ema = Generator(g_output_dim=784).to(device)
    generator_ema.load_state_dict(generator.state_dict())
    generator_ema.eval()
    for p in generator_ema.parameters():
        p.requires_grad_(False)

    generator_params, ema_params = list(generator.parameters()), list(generator_ema.parameters())
    best_ema_fid, history, train_em = torch.inf, [], []
    sampler = IndexSampler(x_train.size(0), device)
    n_critic_samples = args.n_critic * args.batch_size
    start = time.time()
    generator.train()
    critic.train()
    for step in range(1, args.n_steps + 1):
        lr = args.lr * lr_factor(step, args.n_steps, args.warmup_steps, args.min_factor)
        for group in generator_optimizer.param_groups + critic_optimizer.param_groups:
            group["lr"] = lr

        # critic updates (weight clipping), all batches prepared at once since G is fixed meanwhile
        real_batches = x_train[sampler.sample(n_critic_samples)].view(args.n_critic, args.batch_size, -1)
        with torch.no_grad():
            fake_batches = generator(torch.randn(n_critic_samples, 100, device=device)).view(args.n_critic, args.batch_size, -1)
        for x, generated in zip(real_batches, fake_batches):
            c_loss = critic_loss(critic, x, generated)
            critic_optimizer.zero_grad()
            c_loss.backward()
            critic_optimizer.step()
            clip_weights(critic, args.critic_clip)
            train_em.append(c_loss.detach())

        # generator update, then EMA: ema <- ema + (1 - decay) * (g - ema)
        g_loss = generator_loss(critic, generator(torch.randn(args.batch_size, 100, device=device)))
        generator_optimizer.zero_grad()
        g_loss.backward(inputs=generator_params)
        generator_optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, generator_params, 1 - args.ema)

        if step % args.eval_every == 0 or step == args.n_steps:
            generator.eval()
            with torch.no_grad():
                generated = torch.cat([generator(z_eval[i:i + 2000]) for i in range(0, args.n_eval, 2000)])
                generated_ema = torch.cat([generator_ema(z_eval[i:i + 2000]) for i in range(0, args.n_eval, 2000)])
            fid, precision, recall = evaluate_model(clf, feat_real, generated, k=args.k, device=device)
            ema_fid, ema_precision, ema_recall = evaluate_model(clf, feat_real, generated_ema, k=args.k, device=device)
            row = {"step": step,
                   "val_fid": fid, "val_precision": precision, "val_recall": recall,
                   "ema_val_fid": ema_fid, "ema_val_precision": ema_precision, "ema_val_recall": ema_recall,
                   "train_em": -torch.stack(train_em).mean().item(),
                   "lr": lr,
                   "seconds": time.time() - start}
            history.append(row)
            print(json.dumps(row))
            train_em, start = [], time.time()
            generator.train()

            if ema_fid < best_ema_fid:
                best_ema_fid = ema_fid
                torch.save(generator_ema.state_dict(), out_dir.joinpath("best_ema_g.pth"))

            # same keys as train_wgan.py, so that a later run can restart from this one
            torch.save({"step": step,
                        "generator": generator.state_dict(),
                        "generator_ema": generator_ema.state_dict(),
                        "critic": critic.state_dict(),
                        "generator_optimizer": generator_optimizer.state_dict(),
                        "critic_optimizer": critic_optimizer.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                        "best_ema_val_fid": best_ema_fid},
                       out_dir.joinpath("last.pth"))
            with open(out_dir.joinpath("history.json"), "w") as f:
                json.dump(history, f, indent=2)

    torch.save(generator_ema.state_dict(), out_dir.joinpath("final_ema_g.pth"))
    print(f"Best EMA validation FID: {best_ema_fid:.4f}. Final EMA generator -> {out_dir.joinpath('final_ema_g.pth')}")


if __name__ == "__main__":
    main()
