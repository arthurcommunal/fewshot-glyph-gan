"""Few-shot adaptation of the pretrained generator to each new character, K characters at a time (glyphgan.batched).

    python -m scripts.adapt --ckpt runs/pretrain/gp_a/ckpt_200000.pt --mode lora --r 4 --run lora_r4 --part dev

Per character: WGAN-GP between the 10 support images and G_k, starting from the pretrained G (EMA weights) and the
pretrained critic. Critic-side options: FreezeD (--freeze), DiffAugment translation (--shift).
At each --eval_steps step (0 = no adaptation) stores, per character, a pool of P generated images (uint8) and critic
diagnostics; metrics are computed afterwards (scripts/eval_pools.py) so that adaptation and evaluation stay decoupled.
Writes runs/adapt/<run>/{config.json, pools_<step>.pt, diag.jsonl, adapters.pt, grid_<step>.png}.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torchvision

from glyphgan.batched import BatchedGenerator, GroupedCritic, gradient_penalty, translate
from glyphgan.data import episodes
from glyphgan.models import CNNCritic, LAYERS, Z_DIM, new_generator


def to_u8(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def refine_latents(Gb, Cb, z, n_steps: int, step_size: float) -> torch.Tensor:
    """Inference-time refinement inherited from the MNIST project: move each latent along the normalized gradient of
    its (adapted) critic score, z <- z + eta * g / ||g||. The generator weights are unchanged."""
    z = z.clone()
    for _ in range(n_steps):
        z.requires_grad_(True)
        (g,) = torch.autograd.grad(Cb(Gb(z)).sum(), z)
        z = (z + step_size * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)).detach()
    return z


def variants(Gb, Cb, z_pool, step: int, pools: dict, a) -> None:
    """Extra pools at an eval step: truncation (z * psi) and critic-guided refinement of the same latents."""
    for psi in filter(None, a.truncs.split(",")):
        with torch.no_grad():
            pools.setdefault(f"{step}_t{psi}", []).append(to_u8(Gb(float(psi) * z_pool)).cpu())
    for spec in filter(None, a.refine.split(",")):
        n, eta = spec.split("x")
        z = refine_latents(Gb, Cb, z_pool, int(n), float(eta))
        with torch.no_grad():
            pools.setdefault(f"{step}_r{spec}", []).append(to_u8(Gb(z)).cpu())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--part", default="dev")
    p.add_argument("--max_chars", type=int, default=None)
    p.add_argument("--mode", default="lora", choices=["lora", "full"])
    p.add_argument("--r", type=int, default=4)
    p.add_argument("--alpha", type=float, default=None, help="LoRA scale = alpha / r (default alpha = r)")
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--train_bias", action="store_true")
    p.add_argument("--lr_g", type=float, default=1e-3)
    p.add_argument("--lr_c", type=float, default=1e-4)
    p.add_argument("--n_critic", type=int, default=2)
    p.add_argument("--gp", type=float, default=10.0)
    p.add_argument("--drift", type=float, default=1e-3)
    p.add_argument("--freeze", type=int, default=2, help="frozen lower conv blocks of the critic (FreezeD)")
    p.add_argument("--shift", type=int, default=2, help="DiffAugment translation in pixels, 0 = off")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval_steps", default="0,100,250,500,1000")
    p.add_argument("--chunk", type=int, default=128, help="characters adapted simultaneously")
    p.add_argument("--pool", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--truncs", default="0.7", help="extra pools at every eval step > 0 with z * psi (truncation)")
    p.add_argument("--refine", default="10x0.1",
                   help="extra pools at every eval step > 0 with critic-guided latent refinement: n_steps x step_size")
    a = p.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path("runs/adapt") / a.run
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    eval_steps = sorted({int(s) for s in a.eval_steps.split(",") if int(s) <= a.steps})

    ck = torch.load(a.ckpt, map_location="cpu")
    G = new_generator(); G.load_state_dict(ck["G_ema"])
    C = CNNCritic(ck["critic_w"]); C.load_state_dict(ck["critic"])
    eps = episodes(a.part, max_chars=a.max_chars)
    pools: dict[str, list] = {}
    adapters, char_ids = [], []
    diag = open(out / "diag.jsonl", "w")

    for c0 in range(0, len(eps), a.chunk):
        chunk = eps[c0:c0 + a.chunk]
        K = len(chunk)
        torch.manual_seed(a.seed + c0)
        S = torch.stack([e.support for e in chunk]).to(dev)       # (K, 10, 784)
        H = torch.stack([e.heldout for e in chunk]).to(dev)       # (K, 10, 784), diagnostics only
        Gb = BatchedGenerator(G, K, a.mode, a.r, a.alpha, a.layers.split(","), a.train_bias).to(dev)
        Cb = GroupedCritic(C, K, a.freeze).to(dev)
        opt_g = torch.optim.Adam([q for q in Gb.parameters() if q.requires_grad], lr=a.lr_g, betas=(0.0, 0.9))
        opt_c = torch.optim.Adam([q for q in Cb.parameters() if q.requires_grad], lr=a.lr_c, betas=(0.0, 0.9))
        z_pool = torch.randn(K, a.pool, Z_DIM, generator=torch.Generator().manual_seed(a.seed + 10_000 + c0)).to(dev)
        n_sup = S.shape[1]
        t0 = time.time()

        def snapshot(step):
            with torch.no_grad():
                x = Gb(z_pool)
                pools.setdefault(str(step), []).append(to_u8(x).cpu())
                cs, ch, cf = Cb(S).mean(1), Cb(H).mean(1), Cb(x[:, :32]).mean(1)
            for i, e in enumerate(chunk):
                row = {"step": step, "char": e.char, "gap_support_heldout": float(cs[i] - ch[i]),
                       "w_support_fake": float(cs[i] - cf[i]), "sec": time.time() - t0}
                diag.write(json.dumps(row) + "\n")
            diag.flush()
            if step > 0:
                variants(Gb, Cb, z_pool, step, pools, a)
            if c0 == 0:
                grid = torch.cat([S[:8], x[:8, :20]], 1).reshape(-1, 1, 28, 28)
                torchvision.utils.save_image((grid + 1) / 2, out / f"grid_{step}.png", nrow=n_sup + 20)

        if 0 in eval_steps:
            snapshot(0)
        for step in range(1, a.steps + 1):
            for _ in range(a.n_critic):
                real = S.gather(1, torch.randint(n_sup, (K, a.batch, 1), device=dev).expand(-1, -1, S.shape[2]))
                with torch.no_grad():
                    fake = Gb(torch.randn(K, a.batch, Z_DIM, device=dev))
                real, fake = translate(real, a.shift), translate(fake, a.shift)
                sr, sf = Cb(real), Cb(fake)
                loss_c = (sf.mean(1) - sr.mean(1) + a.gp * gradient_penalty(Cb, real, fake)
                          + a.drift * (sr ** 2).mean(1)).sum()
                opt_c.zero_grad(set_to_none=True); loss_c.backward(); opt_c.step()
            fake = translate(Gb(torch.randn(K, a.batch, Z_DIM, device=dev)), a.shift)
            loss_g = -Cb(fake).mean(1).sum()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward(inputs=[q for q in Gb.parameters() if q.requires_grad])
            opt_g.step()
            if step in eval_steps:
                snapshot(step)
        print(json.dumps({"chunk": c0, "K": K, "sec": round(time.time() - t0, 1),
                          "adapter_params": Gb.adapter_numel()}), flush=True)
        if a.mode == "lora":
            adapters.append({k: v.detach().half().cpu() for k, v in Gb.state_dict().items() if k.endswith(("_A", "_B"))})
        char_ids += [e.char for e in chunk]

    for label, chunks in pools.items():
        torch.save({"char_ids": torch.tensor(char_ids), "pool": torch.cat(chunks)}, out / f"pools_{label}.pt")
    if adapters:
        torch.save({"char_ids": char_ids, "chunks": adapters, "scale": Gb.scale}, out / "adapters.pt")
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
