"""Few-shot evaluation protocol shared by every method.

    python -m glyphgan.evaluate --sampler support --part dev --out runs/eval/support_dev.json

Per episode, n_draws draws of n fakes are compared with the 10 heldout reals (n_fake = n_real keeps P/R/coverage
comparable across methods); metrics are averaged over draws, then mean and 95% t-interval are taken across characters.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import stats

from glyphgan.data import Episode, episodes as get_episodes, to_unit
from glyphgan.features import FeatureExtractor
from glyphgan.metrics import copy_stats, identity_acc, prdc

Sampler = Callable[[Episode, int, torch.Generator], torch.Tensor]   # -> (n, 784) in [-1, 1]


def support_sampler(ep: Episode, n: int, g: torch.Generator) -> torch.Tensor:
    """Reference "fresh real data" ceiling: returns the support images (all of them when n == 10)."""
    m = len(ep.support)
    return ep.support[torch.randperm(m, generator=g)[:n] if n <= m else torch.randint(m, (n,), generator=g)]


def alphabet_protos(eps: list[Episode], feats: list[torch.Tensor]) -> dict[int, tuple[torch.Tensor, list[int]]]:
    """alphabet -> (prototypes (C, d) = mean feature per character, list of char ids in the same row order)."""
    by_a: dict[int, list[int]] = defaultdict(list)
    for i, ep in enumerate(eps):
        by_a[ep.alphabet].append(i)
    return {a: (torch.stack([feats[i].mean(0) for i in ix]), [eps[i].char for i in ix]) for a, ix in by_a.items()}


def evaluate(sampler: Sampler, episodes: list[Episode], extractor: FeatureExtractor,
             n_draws: int = 10, n: int = 10, k: int = 3, seed: int = 0) -> tuple[list[dict], dict]:
    """Returns per-character rows (metrics averaged over draws) and their `summarize`. The generator passed to the
    sampler is a CPU torch.Generator seeded by (seed, char, draw)."""
    held = [extractor(ep.heldout) for ep in episodes]
    sup = [extractor(ep.support) for ep in episodes]
    protos = alphabet_protos(episodes, held)
    rows = []
    for ep, hf, sf in zip(episodes, held, sup):
        P, chars = protos[ep.alphabet]
        acc: dict[str, list[float]] = defaultdict(list)
        for d in range(n_draws):
            g = torch.Generator().manual_seed(seed * 1_000_003 + ep.char * 1_000 + d)
            x = sampler(ep, n, g).detach().float().cpu().reshape(n, -1)
            f = extractor(x)
            m = prdc(hf, f, k) | copy_stats(x, ep.support, f.cpu(), sf.cpu())
            m["identity"] = identity_acc(f, P, chars.index(ep.char))
            for key, v in m.items():
                acc[key].append(v)
        rows.append({"char": ep.char, "name": ep.name, "alphabet": ep.alphabet} |
                    {key: float(np.mean(v)) for key, v in acc.items()})
    return rows, summarize(rows)


def evaluate_pools(pools: torch.Tensor, char_ids: list[int] | torch.Tensor, episodes: list[Episode],
                   extractor: FeatureExtractor, n_draws: int = 10, n: int = 10, k: int = 3,
                   seed: int = 0) -> tuple[list[dict], dict]:
    """`evaluate` on stored samples: pools uint8 (N_chars, P, 784), row i = generated pool of character char_ids[i];
    each draw takes n images without replacement from the pool (seeded by the harness generator)."""
    row = {int(c): i for i, c in enumerate(char_ids)}
    missing = [ep.char for ep in episodes if ep.char not in row]
    assert not missing, f"no pool for chars {missing[:5]}..."

    def sampler(ep: Episode, n: int, g: torch.Generator) -> torch.Tensor:
        pool = pools[row[ep.char]]
        return to_unit(pool[torch.randperm(len(pool), generator=g)[:n]])

    return evaluate(sampler, episodes, extractor, n_draws=n_draws, n=n, k=k, seed=seed)


def summarize(rows: list[dict]) -> dict:
    """metric -> {mean, ci (95% t half-width), lo, hi} across characters; plus n_chars and objective."""
    keys = [key for key, v in rows[0].items() if key not in ("char", "name", "alphabet")]
    n = len(rows)
    t = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    out: dict = {"n_chars": n}
    for key in keys:
        v = np.array([r[key] for r in rows], dtype=float)
        mu, hw = float(v.mean()), float(t * v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        out[key] = {"mean": mu, "ci": hw, "lo": mu - hw, "hi": mu + hw}
    if "precision" in out and "coverage" in out:
        out["objective"] = objective(out)
    return out


def objective(summary: dict) -> float:
    """Tuning objective on dev: harmonic mean of mean precision and mean coverage."""
    p, c = summary["precision"]["mean"], summary["coverage"]["mean"]
    return 2 * p * c / (p + c) if p + c > 0 else 0.0


def save_rows(rows: list[dict], summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


def fmt(summary: dict) -> str:
    obj = f" | objective={summary['objective']:.3f}" if "objective" in summary else ""
    return " ".join(f"{k}={v['mean']:.3f}±{v['ci']:.3f}" for k, v in summary.items() if isinstance(v, dict)) + \
        f"{obj} (n={summary['n_chars']})"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sampler", default="support", choices=["support"])
    p.add_argument("--part", default="dev")
    p.add_argument("--clf", default="runs/features/clf.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--n-draws", type=int, default=10)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext, eps = FeatureExtractor.load(a.clf, dev), get_episodes(a.part)
    rows, summ = evaluate(support_sampler, eps, ext, n_draws=a.n_draws)
    print(f"[{a.sampler}/{a.part}]", fmt(summ))
    # fresh-data reference for the copy ratio: heldout reals vs support (what a perfect non-copying sampler gets)
    ref = [copy_stats(ep.heldout, ep.support, ext(ep.heldout).cpu(), ext(ep.support).cpu()) for ep in eps]
    print("[heldout-vs-support copy reference]", fmt(summarize(ref)))
    if a.out:
        save_rows(rows, summ, a.out)
