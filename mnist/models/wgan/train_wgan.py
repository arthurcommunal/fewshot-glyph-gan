import argparse
import json
import time
from pathlib import Path

import torch

from mnist.metrics.classifier import MNISTClassifier
from mnist.metrics.metrics import extract_features
from mnist.model import Generator
from mnist.models.wgan.wgan import (
    Critic,
    EM_estimator,
    IndexSampler,
    clip_weights,
    critic_loss,
    evaluate_model,
    generator_loss,
    init_weights_normal,
    init_weights_uniform,
    subset_to_tensor,
)
from mnist.utils import get_mnist_splits
from mnist.visualize import save_samples


def main() -> None:
    parser = argparse.ArgumentParser("Train a WGAN")
    parser.add_argument("--data_path", type=str, default="data/mnist")
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--critic_clip", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=4)
    parser.add_argument("--k", type=int, default=3, help="For FID, precision and recall estimators" )
    parser.add_argument("--eval_every", type=int, default=4)
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    if args.n_eval > 10000:
        raise ValueError("n_eval must be lower or equal to 10000")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    out_dir = Path("runs").joinpath(Path(args.run_name))
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    else:
        raise FileExistsError(f"{out_dir} already exists")

    # data
    data_path = Path(args.data_path)
    if data_path.exists():
        download = False
    else:
        download = True

    train, val = get_mnist_splits(Path(args.data_path), download=download, seed=42, n_val=10000)
    x_train = subset_to_tensor(train, device)
    x_val = subset_to_tensor(val, device)[:args.n_eval]

    # classifier
    clf_path = Path("runs/clf_local/best.pth")
    clf = MNISTClassifier()
    clf.load_state_dict(torch.load(clf_path, map_location=device))
    clf.to(device)
    feat_real = extract_features(x_val, clf, batch_size=128, device=device)

    # modules
    mnist_dim = 784
    generator = Generator(g_output_dim=mnist_dim).to(device)
    generator.apply(init_weights_normal)

    critic = Critic(input_dim=mnist_dim).to(device)
    critic_init = lambda x: init_weights_uniform(x, args.critic_clip)
    critic.apply(critic_init)

    #optimizers
    generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=args.lr)
    generator_params = list(generator.parameters())

    # scheduler
    generator_scheduler = torch.optim.lr_scheduler.SequentialLR(generator_optimizer,
                                                                [torch.optim.lr_scheduler.LinearLR(generator_optimizer, 1e-2, 1.0, args.warmup_steps),
                                                                 torch.optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, args.n_steps-args.warmup_steps, eta_min=0.01*args.lr)],
                                                                 milestones=[args.warmup_steps])
    critic_scheduler = torch.optim.lr_scheduler.SequentialLR(critic_optimizer,
                                                                [torch.optim.lr_scheduler.LinearLR(critic_optimizer, 1e-2, 1.0, args.warmup_steps),
                                                                 torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, args.n_steps-args.warmup_steps, eta_min=0.01*args.lr)],
                                                                 milestones=[args.warmup_steps])

    best_generator_path = out_dir.joinpath("best_g.pth")
    best_critic_path = out_dir.joinpath("best_c.pth")
    last_path = out_dir.joinpath("last.pth")

    with open(out_dir.joinpath("config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    best_fid, history = torch.inf, []
    train_em = []

    start = time.time()
    sampler = IndexSampler(x_train.size(0), device)
    n_critic_samples = args.n_critic * args.batch_size
    generator.train()
    critic.train()
    for step in range(1, args.n_steps+1):
        # The generator is fixed during the critic updates: prepare all their batches at once
        real_batches = x_train[sampler.sample(n_critic_samples)].view(args.n_critic, args.batch_size, -1)
        with torch.no_grad():
            z = torch.randn(n_critic_samples, 100, device=device)
            fake_batches = generator(z).view(args.n_critic, args.batch_size, -1)

        for x, generated in zip(real_batches, fake_batches):
            c_loss = critic_loss(critic, x, generated)
            critic_optimizer.zero_grad()
            c_loss.backward()
            critic_optimizer.step()
            clip_weights(critic, args.critic_clip)

            train_em.append(c_loss.detach())  # no .item(): it would wait for the GPU at each update

        z = torch.randn(args.batch_size, 100, device=device)
        generated = generator(z)

        g_loss = generator_loss(critic, generated)
        generator_optimizer.zero_grad()
        g_loss.backward(inputs=generator_params)  # skips the unused critic weight gradients
        generator_optimizer.step()

        critic_scheduler.step()
        generator_scheduler.step()

        # eval
        if step % args.eval_every == 0 or step == args.n_steps:
            with torch.no_grad():
                critic.eval()
                generator.eval()
                z = torch.randn(args.n_eval, 100, device=device)
                generated = generator(z)

            val_fid, val_precision, val_recall = evaluate_model(clf, feat_real, generated, k=args.k, device=device)
            val_em = EM_estimator(critic, x_val, generated)

            train_em = -torch.stack(train_em).mean().item()

            row = {"step": step, 
                "val_fid": val_fid, 
                "val_precision": val_precision, 
                "val_recall": val_recall, 
                "val_em_estimator": val_em,
                "train_em": train_em, 
                "lr": generator_scheduler.get_last_lr()[0], 
                "seconds": time.time() - start}
            history.append(row)
            print(json.dumps(row))
            train_em = []
            generator.train()
            critic.train()
            start = time.time()

            if val_fid < best_fid:
                best_fid = val_fid
                torch.save(generator.state_dict(), best_generator_path)
                torch.save(critic.state_dict(), best_critic_path)


            torch.save({"step": step,
                        "generator": generator.state_dict(),
                        "critic": critic.state_dict(),
                        "generator_optimizer": generator_optimizer.state_dict(),
                        "critic_optimizer": critic_optimizer.state_dict(),
                        "generator_scheduler": generator_scheduler.state_dict(),
                        "critic_scheduler": critic_scheduler.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                        "best_val_fid": best_fid},
                        last_path)

            with open(out_dir.joinpath("history.json"), "w") as f:
                json.dump(history, f, indent=2)


    summary = {"best_val_fid": best_fid,
               "best_step": min(history, key=lambda x: x["val_fid"])["step"]}
    with open(out_dir.joinpath("summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best validation FID: {best_fid:.4f} -> {best_generator_path}")

    generator.load_state_dict(torch.load(best_generator_path, map_location=device))
    save_samples(generator, out_dir.joinpath("best_g_samples.png"))


if __name__ == "__main__":
    main()
