"""Batched adaptation == K independent adaptations. Run: python -m tests.test_batched"""
import torch

from glyphgan.batched import BatchedGenerator, GroupedCritic, gradient_penalty, translate
from glyphgan.models import CNNCritic, add_lora, gradient_penalty as gp_single, merge_lora, new_generator

torch.manual_seed(0)
K, B = 3, 5


def test_generator_lora_matches_merged():
    G = new_generator()
    Gb = BatchedGenerator(G, K, "lora", r=2)
    for n, p in Gb.named_parameters():
        p.data.normal_(0, 0.05)
    z = torch.randn(K, B, 100)
    out = Gb(z)
    for k in range(K):
        Gk = new_generator(); Gk.load_state_dict(Gb.merged_state(k))
        assert torch.allclose(out[k], Gk(z[k]), atol=1e-5)


def test_generator_starts_at_pretrained():
    G = new_generator()
    z = torch.randn(K, B, 100)
    for mode in ("lora", "full"):
        assert torch.allclose(BatchedGenerator(G, K, mode, r=4)(z), G(z), atol=1e-5)


def test_lora_module_merge():
    G = new_generator(); Gl = add_lora(G, r=3)
    for n, p in Gl.named_parameters():
        if p.requires_grad: p.data.normal_(0, 0.05)
    z = torch.randn(B, 100)
    assert torch.allclose(Gl(z), merge_lora(Gl)(z), atol=1e-5)


def test_grouped_critic_and_gp():
    Cs = [CNNCritic(8) for _ in range(K)]
    Cb = GroupedCritic(Cs[0], K)
    with torch.no_grad():  # give each group the weights of a different critic
        convs = [[m for m in c.net if isinstance(m, torch.nn.Conv2d)] for c in Cs]
        lins = [m for c in Cs for m in c.net[-1:] if isinstance(m, torch.nn.Linear)]
        for i in range(3):
            getattr(Cb, f"c{i}_w").copy_(torch.cat([cv[i].weight for cv in convs]))
            getattr(Cb, f"c{i}_b").copy_(torch.cat([b for cv in convs if (b := cv[i].bias) is not None]))
        Cb.lin_w.copy_(torch.cat([m.weight for m in lins])); Cb.lin_b.copy_(torch.cat([m.bias for m in lins]))
    x = torch.rand(K, B, 784) * 2 - 1
    s = Cb(x)
    for k in range(K):
        assert torch.allclose(s[k], Cs[k](x[k]), atol=1e-5)
    real, fake = torch.rand(K, B, 784), torch.rand(K, B, 784)
    torch.manual_seed(1); gb = gradient_penalty(Cb, real, fake)
    torch.manual_seed(1); a = torch.rand(K, B, 1)
    for k in range(K):
        xh = (a[k] * real[k] + (1 - a[k]) * fake[k]).requires_grad_(True)
        (g,) = torch.autograd.grad(Cs[k](xh).sum(), xh)
        assert torch.allclose(gb[k], ((g.norm(dim=1) - 1) ** 2).mean(), atol=1e-4)


def test_frozen_shared_layers():
    C = CNNCritic(8)
    x = torch.rand(K, B, 784) * 2 - 1
    ref = torch.stack([C(x[k]) for k in range(K)])
    for f in (0, 1, 2, 3):
        Cb = GroupedCritic(C, K, freeze=f)
        assert torch.allclose(Cb(x), ref, atol=1e-5), f
        real, fake = torch.rand(K, B, 784), torch.rand(K, B, 784)
        torch.manual_seed(2); g0 = gradient_penalty(GroupedCritic(C, K, 0), real, fake)
        torch.manual_seed(2); g1 = gradient_penalty(Cb, real, fake)
        assert torch.allclose(g0, g1, atol=1e-4), f


def test_translate():
    x = torch.rand(K, B, 784) * 2 - 1
    assert torch.equal(translate(x, 0), x)
    y = translate(x, 2)
    assert y.shape == x.shape and y.min() >= -1 and y.max() <= 1
    # ink is preserved for a centred glyph (no clipping by the shift)
    img = -torch.ones(1, 1, 28, 28); img[..., 10:18, 10:18] = 1
    t = translate(img.view(1, 1, 784).expand(4, 8, 784).contiguous(), 2)
    assert torch.allclose(((t + 1) / 2).sum(-1), torch.full((4, 8), 64.0))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
