"""Headline comparison on test: paired bootstrap over characters (same 314 characters for every method).

    python -m scripts.headline A=runs/adapt/t2_full_lr1e-4/eval_500_r10x0.1.json B=runs/adapt/aug_feat09_test/eval_0.json

For each method: mean and 95% CI of precision, coverage, identity, copy_feat and the objective (harmonic mean of the
mean precision and mean coverage); then the paired difference A - B with a 95% bootstrap CI (10k resamples of
characters, the objective recomputed on each resample).
"""
import json
import sys

import numpy as np

KEYS = ["precision", "coverage", "identity", "copy_feat", "density", "recall"]


def load(path):
    rows = {r["char"]: r for r in json.load(open(path))["rows"]}
    return rows


def obj(p, c):
    return 2 * p * c / (p + c)


def stats(rows, chars, idx):
    out = {k: np.array([rows[c][k] for c in chars])[idx].mean(axis=-1) for k in KEYS}
    out["objective"] = obj(out["precision"], out["coverage"])
    return out


def main():
    named = [a.split("=", 1) for a in sys.argv[1:]]
    data = {n: load(p) for n, p in named}
    chars = sorted(set.intersection(*[set(d) for d in data.values()]))
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(chars), size=(10_000, len(chars)))
    full = np.arange(len(chars))[None]
    res = {}
    print(f"n_chars = {len(chars)}")
    for n, d in data.items():
        point, bs = stats(d, chars, full), stats(d, chars, boot)
        res[n] = bs
        print(f"{n:6s} " + "  ".join(f"{k}={point[k][0]:.3f} [{np.percentile(bs[k], 2.5):.3f},{np.percentile(bs[k], 97.5):.3f}]"
                                       for k in ["objective"] + KEYS))
    names = list(data)
    for b in names[1:]:
        a = names[0]
        print(f"{a}-{b} " + "  ".join(
            f"d_{k}={np.mean(res[a][k] - res[b][k]):+.3f} [{np.percentile(res[a][k] - res[b][k], 2.5):+.3f},"
            f"{np.percentile(res[a][k] - res[b][k], 97.5):+.3f}]" for k in ["objective", "precision", "coverage", "identity", "copy_feat"]))


if __name__ == "__main__":
    main()
