"""Prior-only baseline: GAN inversion in the latent space of the frozen pretrained generator (no weight change).

    python -m scripts.invert --ckpt runs/pretrain/gp_a/ckpt_30000.pt --run invert_a --part dev

Per character (K characters at a time, all on GPU): each of the 10 support images x_j is inverted into z_j by Adam
on  ||G(z) - x_j||^2 / 784  +  feat * (1 - <f(G(z)), f(x_j)>)  +  prior * (||z|| / sqrt(d) - 1)^2,  with f the
frozen metric embedding (unit norm) and several random restarts per support (the lowest final loss is kept).
Pool of P samples: G(z_i + sigma * eps), i uniform over the 10 inverted latents, eps ~ N(0, I); the same (i, eps)
are reused for every sigma. Answers "is the pretrained latent space already enough, without adapting weights?".
Writes runs/adapt/<run>/{config.json, pools_<k>.pt (k = sigma index, see config["sigmas"]), recon.json, grid.png},
evaluated with `python -m scripts.eval_pools runs/adapt/<run>`.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torchvision

from glyphgan.data import episodes
from glyphgan.features import FeatureExtractor
from glyphgan.models import Z_DIM, new_generator


def to_u8(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def invert(G, embed, S, restarts, steps, lr, feat, prior, gen):
    """S (K, n, 784) -> best z (K, n, d), per-support MSE (K, n), per-support feature cosine (K, n)."""
    K, n, D = S.shape
    target = S.unsqueeze(2).expand(K, n, restarts, D).reshape(-1, D)
    f_target = embed(target).detach() if feat > 0 else None
    z = torch.randn(K * n * restarts, Z_DIM, generator=gen, device=S.device).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t / steps)))

    def losses(z):
        x = G(z)
        mse = (x - target).pow(2).mean(1)
        cos = (embed(x) * f_target).sum(1) if feat > 0 else torch.zeros_like(mse)
        pr = (z.norm(dim=1) / math.sqrt(Z_DIM) - 1).pow(2)
        return mse + feat * (1 - cos) + prior * pr, mse, cos

    for _ in range(steps):
        loss = losses(z)[0].sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        tot, mse, cos = (t.view(K, n, restarts) for t in losses(z))
        best = tot.argmin(2)                                                      # (K, n)
        pick = lambda t: t.gather(2, best.unsqueeze(2)).squeeze(2)                # noqa: E731
        zb = z.detach().view(K, n, restarts, Z_DIM).gather(2, best[..., None, None].expand(K, n, 1, Z_DIM)).squeeze(2)
        return zb, pick(mse), pick(cos)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--part", default="dev")
    p.add_argument("--max_chars", type=int, default=None)
    p.add_argument("--sigmas", default="0,0.1,0.25,0.5,1.0")
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--feat", type=float, default=0.1, help="weight of the feature loss (0 = pixel MSE only)")
    p.add_argument("--prior", type=float, default=0.1, help="weight of (||z||/sqrt(d) - 1)^2")
    p.add_argument("--chunk", type=int, default=64, help="characters inverted simultaneously")
    p.add_argument("--pool", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out = Path("runs/adapt") / a.run
    out.mkdir(parents=True, exist_ok=False)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    cfg = {**vars(a), "method": "invert", "sigmas": {str(k): s for k, s in enumerate(sigmas)}}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    ck = torch.load(a.ckpt, map_location="cpu")
    G = new_generator(); G.load_state_dict(ck["G_ema"])
    G = G.to(dev).eval().requires_grad_(False)
    embed = FeatureExtractor.load("runs/features/clf.pt", dev).net.embed
    eps_list = episodes(a.part, max_chars=a.max_chars)
    pools = [[] for _ in sigmas]
    char_ids, mses, coss, znorms = [], [], [], []
    t0 = time.time()

    for c0 in range(0, len(eps_list), a.chunk):
        chunk = eps_list[c0:c0 + a.chunk]
        K = len(chunk)
        gen = torch.Generator(device=dev).manual_seed(a.seed + c0)
        S = torch.stack([e.support for e in chunk]).to(dev)                     # (K, 10, 784)
        zb, mse, cos = invert(G, embed, S, a.restarts, a.steps, a.lr, a.feat, a.prior, gen)
        n_sup = S.shape[1]
        g2 = torch.Generator(device=dev).manual_seed(a.seed + 10_000 + c0)
        idx = torch.randint(n_sup, (K, a.pool), generator=g2, device=dev)
        noise = torch.randn(K, a.pool, Z_DIM, generator=g2, device=dev)
        z0 = zb.gather(1, idx.unsqueeze(2).expand(-1, -1, Z_DIM))
        with torch.no_grad():
            xs = [G(z0 + s * noise) for s in sigmas]
            for k, x in enumerate(xs):
                pools[k].append(to_u8(x).cpu())
            if c0 == 0:
                R = G(zb)
                rows = []
                for i in range(min(6, K)):
                    rows += [S[i], R[i]] + [x[i, :n_sup] for x in xs]
                grid = torch.cat(rows).reshape(-1, 1, 28, 28)
                torchvision.utils.save_image((grid + 1) / 2, out / "grid.png", nrow=n_sup, padding=1)
        char_ids += [e.char for e in chunk]
        mses.append(mse.cpu()); coss.append(cos.cpu()); znorms.append(zb.norm(dim=2).cpu())
        print(json.dumps({"chunk": c0, "K": K, "recon_mse": round(float(mse.mean()), 4),
                          "feat_cos": round(float(cos.mean()), 4), "sec": round(time.time() - t0, 1)}), flush=True)

    for k in range(len(sigmas)):
        torch.save({"char_ids": torch.tensor(char_ids), "pool": torch.cat(pools[k])}, out / f"pools_{k}.pt")
    mse, cos, zn = torch.cat(mses), torch.cat(coss), torch.cat(znorms)
    recon = {"recon_mse": float(mse.mean()), "recon_mse_per_char_std": float(mse.mean(1).std()),
             "feat_cos": float(cos.mean()), "z_norm": float(zn.mean()), "n_chars": len(char_ids),
             "sec": round(time.time() - t0, 1)}
    (out / "recon.json").write_text(json.dumps(recon, indent=2))
    print(json.dumps(recon), "done", out, flush=True)


if __name__ == "__main__":
    main()
