"""K independent few-shot adaptations run as ONE batched computation on one GPU.

Each symbol k has its own generator update (LoRA factors A_k, B_k, or full weights W_k) and its own critic copy.
  - generator: the frozen base W0 is shared, so h W0^T is one big matmul; the per-symbol low-rank term is a bmm.
  - critic: K copies of the pretrained CNN critic = grouped convolutions (groups=K), symbol k on channel group k.
The total loss is the SUM of the K per-symbol losses, so each symbol's parameters receive exactly the gradient of its
own loss, and Adam (element-wise) on the stacked parameters equals K independent Adams. Mathematically identical to
running the K adaptations one after the other, but ~K times fewer kernel launches.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from glyphgan.models import CNNCritic, LAYERS


class BatchedGenerator(nn.Module):
    """mode='lora': W_k = W0 + (alpha/r) B_k A_k ; mode='full': W_k free (initialised at W0)."""

    def __init__(self, G: nn.Module, K: int, mode: str = "lora", r: int = 4, alpha: float | None = None,
                 layers=LAYERS, train_bias: bool = False):
        super().__init__()
        self.K, self.mode, self.layers = K, mode, list(layers)
        self.scale = (alpha if alpha is not None else r) / r if mode == "lora" else 0.0
        for name in LAYERS:
            lin = getattr(G, name)
            W0, b0 = lin.weight.detach().clone(), lin.bias.detach().clone()
            self.register_buffer(f"{name}_W0", W0)
            adapt = name in self.layers
            if mode == "full" and adapt:
                setattr(self, f"{name}_W", nn.Parameter(W0.expand(K, -1, -1).clone()))
            if mode == "lora" and adapt:
                out_f, in_f = W0.shape
                bound = 1 / math.sqrt(in_f)  # = kaiming_uniform(a=sqrt(5)), nn.Linear / peft default for A
                setattr(self, f"{name}_A", nn.Parameter(torch.empty(K, r, in_f).uniform_(-bound, bound)))
                setattr(self, f"{name}_B", nn.Parameter(torch.zeros(K, out_f, r)))
            if (train_bias or mode == "full") and adapt:
                setattr(self, f"{name}_b", nn.Parameter(b0.expand(K, -1).clone()))
            else:
                self.register_buffer(f"{name}_b", b0.expand(K, -1).clone())

    def _linear(self, name: str, h: torch.Tensor) -> torch.Tensor:  # h (K, B, in)
        b = getattr(self, f"{name}_b").unsqueeze(1)
        if self.mode == "full" and name in self.layers:
            return torch.baddbmm(b, h, getattr(self, f"{name}_W").transpose(1, 2))
        out = h @ getattr(self, f"{name}_W0").t() + b
        if self.mode == "lora" and name in self.layers:
            A, B = getattr(self, f"{name}_A"), getattr(self, f"{name}_B")
            out = out + self.scale * torch.bmm(torch.bmm(h, A.transpose(1, 2)), B.transpose(1, 2))
        return out

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # z (K, B, 100) -> (K, B, 784)
        h = z
        for i, name in enumerate(LAYERS):
            h = self._linear(name, h)
            h = torch.tanh(h) if i == len(LAYERS) - 1 else F.leaky_relu(h, 0.2)
        return h

    @torch.no_grad()
    def merged_state(self, k: int) -> dict:
        """Plain Generator state_dict for symbol k: the deployed model keeps the original architecture."""
        sd = {}
        for name in LAYERS:
            W = getattr(self, f"{name}_W0").clone()
            if self.mode == "full" and name in self.layers:
                W = getattr(self, f"{name}_W")[k].clone()
            if self.mode == "lora" and name in self.layers:
                W += self.scale * getattr(self, f"{name}_B")[k] @ getattr(self, f"{name}_A")[k]
            sd[f"{name}.weight"], sd[f"{name}.bias"] = W, getattr(self, f"{name}_b")[k].clone()
        return sd

    def adapter_numel(self) -> int:
        """Per-symbol stored parameters."""
        return sum(p[0].numel() for p in self.parameters() if p.requires_grad)


class GroupedCritic(nn.Module):
    """K copies of a pretrained CNNCritic. freeze = number of lower conv blocks kept frozen (FreezeD, Mo et al. 2020).

    Frozen blocks are identical for every symbol: they run as ONE ordinary convolution on the K*B images (cuDNN, fast
    double backward for the gradient penalty). Adapted blocks run as grouped convolutions (groups=K, symbol k on
    channel group k). An im2col + bmm version was tried and was 2x slower than grouped convolutions."""

    def __init__(self, C: CNNCritic, K: int, freeze: int = 0):
        super().__init__()
        self.K, self.w, self.freeze, self.size = K, C.w, freeze, getattr(C, "size", 28)
        convs = [m for m in C.net if isinstance(m, nn.Conv2d)]
        lin = C.net[-1]
        self.specs = [(c.stride[0], c.padding[0]) for c in convs]
        for i, c in enumerate(convs):
            if i < freeze:   # shared, frozen
                self.register_buffer(f"c{i}_w", c.weight.detach().clone())
                self.register_buffer(f"c{i}_b", c.bias.detach().clone())
            else:            # one copy per symbol, stacked along the output channels
                setattr(self, f"c{i}_w", nn.Parameter(c.weight.detach().repeat(K, 1, 1, 1).clone()))
                setattr(self, f"c{i}_b", nn.Parameter(c.bias.detach().repeat(K).clone()))
        self.lin_w = nn.Parameter(lin.weight.detach().expand(K, -1).clone())   # (K, 4w*16)
        self.lin_b = nn.Parameter(lin.bias.detach().expand(K).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (K, B, 784) -> scores (K, B)
        K, B = x.shape[:2]
        h = x.reshape(K * B, 1, self.size, self.size)
        for i in range(self.freeze):
            s, p = self.specs[i]
            h = F.leaky_relu(F.conv2d(h, getattr(self, f"c{i}_w"), getattr(self, f"c{i}_b"), s, p), 0.2)
        h = h.reshape(K, B, *h.shape[1:]).transpose(0, 1).reshape(B, K * h.shape[1], *h.shape[2:])
        for i in range(self.freeze, len(self.specs)):
            s, p = self.specs[i]
            h = F.leaky_relu(F.conv2d(h, getattr(self, f"c{i}_w"), getattr(self, f"c{i}_b"), s, p, groups=K), 0.2)
        h = h.reshape(B, K, -1)                     # group k's channels are contiguous: = nn.Flatten of critic k
        return (torch.einsum("bkf,kf->bk", h, self.lin_w) + self.lin_b).t()


def gradient_penalty(critic: GroupedCritic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Per-symbol WGAN-GP term, shape (K,). Scores of different symbols are independent, so the gradient of their sum
    w.r.t. x_hat[k] is the gradient of critic k alone."""
    a = torch.rand(real.shape[0], real.shape[1], 1, device=real.device)
    x_hat = (a * real + (1 - a) * fake).requires_grad_(True)
    (g,) = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)
    return ((g.norm(2, dim=2) - 1) ** 2).mean(1)


def translate(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    """DiffAugment-style random integer translation (Zhao et al. 2020), differentiable, background -1 padding.
    x (K, B, 784)."""
    if max_shift <= 0:
        return x
    K, B, D = x.shape
    n = int(round(D ** 0.5))
    img = x.reshape(K * B, 1, n, n) + 1                       # background -> 0 so that zero padding is background
    pad = F.pad(img, (max_shift,) * 4)
    sx = torch.randint(0, 2 * max_shift + 1, (K * B,), device=x.device)
    sy = torch.randint(0, 2 * max_shift + 1, (K * B,), device=x.device)
    ar = torch.arange(n, device=x.device)
    rows = (sy[:, None] + ar[None]).view(-1, 1, n, 1).expand(-1, 1, n, n)
    cols = (sx[:, None] + ar[None]).view(-1, 1, 1, n).expand(-1, 1, n, n)
    # vectorised crop: select the 28 rows, then the 28 columns
    pr = pad.gather(2, rows[..., :1].expand(-1, 1, n, pad.shape[3]))           # (KB, 1, 28, W+2s)
    out = pr.gather(3, cols)                                                   # (KB, 1, 28, 28)
    return (out - 1).reshape(K, B, D)
