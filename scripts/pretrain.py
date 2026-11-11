"""WGAN-GP pretraining of the fixed generator on Omniglot background (all 964 characters, unconditional).

    python -m scripts.pretrain --run gp_base --steps 120000

Writes runs/pretrain/<run>/{config.json, log.jsonl, samples_<step>.png, ckpt_<step>.pt, last.pt}.
ckpt_<step>.pt = {G, G_ema, critic} state dicts; checkpoint selection is done afterwards (scripts/select_pretrain.py)
with the Omniglot feature extractor, which is trained in parallel.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torchvision

from glyphgan.data import load_split
from glyphgan.models import CNNCritic, Z_DIM, ema_update, gradient_penalty, new_generator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--steps", type=int, default=120_000, help="generator updates")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_c", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--gp", type=float, default=10.0)
    p.add_argument("--drift", type=float, default=1e-3, help="eps * E[C(x)^2], keeps critic scores centred")
    p.add_argument("--critic_w", type=int, default=64)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--ckpt_every", type=int, default=10_000)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--size", type=int, default=28, help="28 (fixed generator) or 56 (output layer widened)")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out = Path("runs/pretrain") / a.run
    out.mkdir(parents=True, exist_ok=a.resume)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    x = load_split(a.data, "background", a.size).x.to(dev)
    G = new_generator(a.size * a.size).to(dev)
    G_ema = copy.deepcopy(G).eval()
    C = CNNCritic(a.critic_w, a.size).to(dev)
    opt_g = torch.optim.Adam(G.parameters(), lr=a.lr_g, betas=(a.beta1, a.beta2))
    opt_c = torch.optim.Adam(C.parameters(), lr=a.lr_c, betas=(a.beta1, a.beta2))
    start = 1
    if a.resume and (out / "last.pt").exists():
        s = torch.load(out / "last.pt", map_location=dev)
        G.load_state_dict(s["G"]); G_ema.load_state_dict(s["G_ema"]); C.load_state_dict(s["critic"])
        opt_g.load_state_dict(s["opt_g"]); opt_c.load_state_dict(s["opt_c"]); start = s["step"] + 1
    z_fixed = torch.randn(128, Z_DIM, generator=torch.Generator().manual_seed(123)).to(dev)

    log = open(out / "log.jsonl", "a")
    acc = {"w": 0.0, "gp": 0.0, "n": 0}
    t0 = time.time()
    for step in range(start, a.steps + 1):
        for _ in range(a.n_critic):
            real = x[torch.randint(len(x), (a.batch,), device=dev)]
            with torch.no_grad():
                fake = G(torch.randn(a.batch, Z_DIM, device=dev))
            s_real, s_fake = C(real), C(fake)
            gp = gradient_penalty(C, real, fake)
            loss_c = s_fake.mean() - s_real.mean() + a.gp * gp + a.drift * (s_real ** 2).mean()
            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            opt_c.step()
        acc["w"] += (s_real.mean() - s_fake.mean()).detach(); acc["gp"] += gp.detach(); acc["n"] += 1

        loss_g = -C(G(torch.randn(a.batch, Z_DIM, device=dev))).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward(inputs=list(G.parameters()))
        opt_g.step()
        ema_update(G_ema, G, a.ema)

        if step % a.log_every == 0:
            row = {"step": step, "w_est": float(acc["w"] / acc["n"]), "gp": float(acc["gp"] / acc["n"]),
                   "sec_per_step": (time.time() - t0) / a.log_every}
            print(json.dumps(row), flush=True); log.write(json.dumps(row) + "\n"); log.flush()
            acc = {"w": 0.0, "gp": 0.0, "n": 0}; t0 = time.time()
        if step % a.ckpt_every == 0 or step == a.steps:
            with torch.no_grad():
                torchvision.utils.save_image((G_ema(z_fixed).view(-1, 1, a.size, a.size) + 1) / 2, out / f"samples_{step}.png", nrow=16)
            torch.save({"G": G.state_dict(), "G_ema": G_ema.state_dict(), "critic": C.state_dict(),
                        "critic_w": a.critic_w, "size": a.size, "step": step}, out / f"ckpt_{step}.pt")
            torch.save({"G": G.state_dict(), "G_ema": G_ema.state_dict(), "critic": C.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_c": opt_c.state_dict(), "step": step}, out / "last.pt")


if __name__ == "__main__":
    main()
