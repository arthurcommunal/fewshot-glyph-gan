"""Sample-quality metrics on small sets (pure functions on tensors).

All feature-space distances are Euclidean on the L2-normalized embeddings of `glyphgan.features`.
"""
import torch


def _kth_nn_radius(x: torch.Tensor, k: int) -> torch.Tensor:
    """Distance of each row of x to its k-th nearest neighbour inside x (self excluded)."""
    d = torch.cdist(x, x)
    return d.kthvalue(k + 1, dim=1).values                               # +1: the 0 self-distance is the 1st


def prdc(real_feat: torch.Tensor, fake_feat: torch.Tensor, k: int = 3) -> dict[str, float]:
    """Precision/recall (Kynkaanniemi et al. 2019) and density/coverage (Naeem et al. 2020) with k-NN balls."""
    real_feat, fake_feat = real_feat.float(), fake_feat.float()
    r_real, r_fake = _kth_nn_radius(real_feat, k), _kth_nn_radius(fake_feat, k)
    d = torch.cdist(real_feat, fake_feat)                                # (n_real, n_fake)
    in_real = d <= r_real[:, None]                                       # fake j inside the ball of real i
    return {
        "precision": in_real.any(0).float().mean().item(),
        "recall": (d <= r_fake[None, :]).any(1).float().mean().item(),
        "density": (in_real.float().sum(0) / k).mean().item(),
        "coverage": (d.min(1).values <= r_real).float().mean().item(),
    }


def _copy_ratio(gen: torch.Tensor, support: torch.Tensor) -> float:
    """median NN distance gen->support / median leave-one-out NN distance within support."""
    gen, support = gen.flatten(1).float(), support.flatten(1).float()
    d_gen = torch.cdist(gen, support).min(1).values
    d_loo = _kth_nn_radius(support, 1)
    return (d_gen.median() / d_loo.median().clamp_min(1e-12)).item()


def copy_stats(gen_x: torch.Tensor, support_x: torch.Tensor,
               gen_feat: torch.Tensor, support_feat: torch.Tensor) -> dict[str, float]:
    """Memorization ratio in pixel and feature space: ~1 for fresh real data, << 1 means copying the support."""
    return {"copy_pix": _copy_ratio(gen_x, support_x), "copy_feat": _copy_ratio(gen_feat, support_feat)}


def identity_acc(fake_feat: torch.Tensor, protos: torch.Tensor, target: int) -> float:
    """Fraction of fakes whose nearest prototype (cosine) is `target` (row index in protos)."""
    sim = fake_feat.float() @ torch.nn.functional.normalize(protos.float(), dim=1).T
    return (sim.argmax(1) == target).float().mean().item()
