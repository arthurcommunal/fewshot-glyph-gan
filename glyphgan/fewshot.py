"""Few-shot adaptation of ONE new symbol, for interactive use (scripts/demo.py).

Same loop as scripts/adapt.py with K = 1 and the configuration selected on dev: LoRA r = 16 on fc1..fc4, lr 1e-4,
1500 steps, FreezeD 2, DiffAugment translation 2 px, then critic-guided latent refinement (10 steps of 0.1).
"""
from dataclasses import dataclass
from typing import Callable

import torch

from glyphgan.batched import BatchedGenerator, GroupedCritic, gradient_penalty, translate
from glyphgan.models import CNNCritic, Z_DIM, new_generator


@dataclass
class Config:
    r: int = 16
    lr_g: float = 1e-4
    lr_c: float = 1e-4
    steps: int = 1500
    n_critic: int = 2
    gp: float = 10.0
    drift: float = 1e-3
    freeze: int = 2
    shift: int = 2
    batch: int = 16
    refine_steps: int = 10
    refine_eta: float = 0.1


def load_pretrained(path: str):
    ck = torch.load(path, map_location="cpu")
    size = ck.get("size", 28)
    G = new_generator(size * size); G.load_state_dict(ck["G_ema"]); G.eval()
    C = CNNCritic(ck["critic_w"], size); C.load_state_dict(ck["critic"])
    return G, C


def adapt_symbol(G, C, support: torch.Tensor, cfg: Config = Config(), seed: int = 0,
                 callback: Callable[[int, int], None] | None = None):
    """support (n, 784) in [-1, 1], on the device to use -> (Gb, Cb), adapted generator and critic of this symbol (K = 1)."""
    torch.manual_seed(seed)
    dev = support.device
    S = support.unsqueeze(0)                                        # (1, n, 784)
    n = S.shape[1]
    Gb = BatchedGenerator(G, 1, "lora", cfg.r).to(dev)
    Cb = GroupedCritic(C, 1, cfg.freeze).to(dev)
    opt_g = torch.optim.Adam([q for q in Gb.parameters() if q.requires_grad], lr=cfg.lr_g, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam([q for q in Cb.parameters() if q.requires_grad], lr=cfg.lr_c, betas=(0.0, 0.9))
    for step in range(1, cfg.steps + 1):
        for _ in range(cfg.n_critic):
            real = S[:, torch.randint(n, (cfg.batch,), device=dev)]
            with torch.no_grad():
                fake = Gb(torch.randn(1, cfg.batch, Z_DIM, device=dev))
            real, fake = translate(real, cfg.shift), translate(fake, cfg.shift)
            sr, sf = Cb(real), Cb(fake)
            loss_c = (sf.mean(1) - sr.mean(1) + cfg.gp * gradient_penalty(Cb, real, fake)
                      + cfg.drift * (sr ** 2).mean(1)).sum()
            opt_c.zero_grad(set_to_none=True); loss_c.backward(); opt_c.step()
        loss_g = -Cb(translate(Gb(torch.randn(1, cfg.batch, Z_DIM, device=dev)), cfg.shift)).mean(1).sum()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward(inputs=[q for q in Gb.parameters() if q.requires_grad])
        opt_g.step()
        if callback is not None:
            callback(step, cfg.steps)
    return Gb, Cb


def sample(Gb, Cb, n: int, cfg: Config = Config(), seed: int = 1, refine: bool = True) -> torch.Tensor:
    """n samples (n, 784) in [-1, 1]; refine = critic-guided latent refinement z <- z + eta g/||g||."""
    z = torch.randn(1, n, Z_DIM, generator=torch.Generator().manual_seed(seed)).to(next(Gb.buffers()).device)
    if refine:
        for _ in range(cfg.refine_steps):
            z.requires_grad_(True)
            (g,) = torch.autograd.grad(Cb(Gb(z)).sum(), z)
            z = (z + cfg.refine_eta * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)).detach()
    with torch.no_grad():
        return Gb(z)[0]
