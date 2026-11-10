"""Synthetic sanity tests for glyphgan.metrics / glyphgan.evaluate (pytest-compatible; also `python -m tests.test_metrics`)."""
import torch
import torch.nn.functional as F

from glyphgan.data import Episode
from glyphgan.evaluate import evaluate, evaluate_pools, objective, summarize, support_sampler
from glyphgan.features import FeatureExtractor, FeatureNet, augment
from glyphgan.metrics import copy_stats, identity_acc, prdc


def _cloud(n: int = 10, d: int = 16, seed: int = 0, shift: float = 0.0) -> torch.Tensor:
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed)) + shift


def test_prdc_identical() -> None:
    x = _cloud()
    m = prdc(x, x.clone(), k=3)
    assert m["precision"] == m["recall"] == m["coverage"] == 1.0, m
    assert m["density"] >= 1.0, m                     # each point is in its own ball + its k neighbours' balls


def test_prdc_disjoint() -> None:
    m = prdc(_cloud(), _cloud(seed=1, shift=100.0), k=3)
    assert all(v == 0.0 for v in m.values()), m


def test_prdc_same_distribution_reasonable() -> None:
    m = prdc(_cloud(200, 4), _cloud(200, 4, seed=1), k=5)
    assert m["precision"] > 0.7 and m["recall"] > 0.7 and m["coverage"] > 0.7, m


def test_copy_stats() -> None:
    sup, fresh = _cloud(10, 784), _cloud(10, 784, seed=1)
    copies = copy_stats(sup.clone(), sup, sup.clone(), sup)
    assert copies["copy_pix"] == 0.0 and copies["copy_feat"] == 0.0, copies
    real = copy_stats(fresh, sup, fresh, sup)
    assert 0.8 < real["copy_pix"] < 1.2, real


def test_identity_acc() -> None:
    protos = torch.eye(4)
    fake = F.normalize(protos[[2, 2, 2, 1]] + 0.1, dim=1)
    assert identity_acc(fake, protos, 2) == 0.75


def test_evaluate_pipeline() -> None:
    torch.manual_seed(0)
    ext = FeatureExtractor(FeatureNet(5), "cpu")      # random weights: only checks shapes / plumbing
    eps = [Episode(c, f"A/c{c}", c // 2, torch.rand(10, 784) * 2 - 1, torch.rand(10, 784) * 2 - 1) for c in range(4)]
    rows, summ = evaluate(support_sampler, eps, ext, n_draws=2)
    assert len(rows) == 4 and summ["n_chars"] == 4
    assert summ["copy_pix"]["mean"] == 0.0 and 0.0 <= objective(summ) <= 1.0
    assert summarize(rows)["precision"]["lo"] <= summ["precision"]["mean"]
    assert augment(eps[0].support).shape == (10, 784)
    pools = ((torch.stack([ep.support for ep in eps]) + 1) * 127.5).round().to(torch.uint8)   # pool = support
    rows_p, summ_p = evaluate_pools(pools, [ep.char for ep in eps], eps, ext, n_draws=2)
    assert abs(summ_p["precision"]["mean"] - summ["precision"]["mean"]) < 0.2 and summ_p["copy_pix"]["mean"] < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
