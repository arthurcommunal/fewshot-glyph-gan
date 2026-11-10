"""Generator (fixed architecture, from the MNIST project), CNN critic, LoRA.

LoRA (Hu et al., 2021) on a frozen linear layer W (out x in):
    W' = W + (alpha / r) * B @ A,   A (r x in) random,  B (out x r) = 0 at init,
so the adapted generator starts exactly at the pretrained one. After adaptation the update is merged into W:
the deployed generator has the original architecture and parameter count; only A, B are stored per symbol.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from mnist.model import Generator  # noqa: F401  (the fixed 100-256-512-1024-784 MLP, never modified)

Z_DIM = 100
X_DIM = 784


def new_generator() -> Generator:
    g = Generator(g_output_dim=X_DIM)
    for m in g.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            nn.init.zeros_(m.bias)
    return g


class CNNCritic(nn.Module):
    """DCGAN-like critic without normalization (compatible with the gradient penalty). Free to choose: only the
    generator architecture is imposed."""

    def __init__(self, w: int = 64):
        super().__init__()
        self.w = w
        self.net = nn.Sequential(
            nn.Conv2d(1, w, 4, 2, 1), nn.LeakyReLU(0.2),          # 28 -> 14
            nn.Conv2d(w, 2 * w, 4, 2, 1), nn.LeakyReLU(0.2),      # 14 -> 7
            nn.Conv2d(2 * w, 4 * w, 3, 2, 1), nn.LeakyReLU(0.2),  # 7 -> 4
            nn.Flatten(), nn.Linear(4 * w * 16, 1))

    def forward(self, x):
        return self.net(x.view(-1, 1, 28, 28)).squeeze(1)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Gulrajani et al. 2017: E[(||grad_x C(x_hat)|| - 1)^2] on random interpolates."""
    a = torch.rand(real.size(0), 1, device=real.device)
    x_hat = (a * real + (1 - a) * fake).requires_grad_(True)
    (grad,) = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, train_bias: bool = False):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        for p in base.parameters():
            p.requires_grad_(False)
        if train_bias:
            base.bias.requires_grad_(True)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t()) @ self.B.t()

    def merged(self) -> nn.Linear:
        lin = copy.deepcopy(self.base)
        with torch.no_grad():
            lin.weight += self.scale * self.B @ self.A
        for p in lin.parameters():
            p.requires_grad_(False)
        return lin


LAYERS = ("fc1", "fc2", "fc3", "fc4")


def add_lora(g: Generator, r: int, alpha: float | None = None, layers=LAYERS, train_bias: bool = False) -> Generator:
    """Returns a deep copy of g with LoRA on `layers` (all other weights frozen). alpha defaults to r (scale 1)."""
    g = copy.deepcopy(g)
    for p in g.parameters():
        p.requires_grad_(False)
    for name in layers:
        setattr(g, name, LoRALinear(getattr(g, name), r, alpha if alpha is not None else r, train_bias))
    return g


def merge_lora(g: Generator) -> Generator:
    """Folds every LoRALinear back into a plain nn.Linear: same architecture as the original Generator."""
    g = copy.deepcopy(g)
    for name in LAYERS:
        m = getattr(g, name)
        if isinstance(m, LoRALinear):
            setattr(g, name, m.merged())
    return g


def lora_state(g: Generator) -> dict:
    """Only the per-symbol parameters (what would be stored/shipped for one new symbol)."""
    return {k: v.detach().cpu() for k, v in g.state_dict().items() if k.endswith((".A", ".B"))}


def n_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1 - decay)
