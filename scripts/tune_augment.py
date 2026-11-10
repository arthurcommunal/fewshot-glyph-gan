"""Random search over the strengths of the augmentation baseline, on dev episodes.

    python -m scripts.tune_augment --n-random 70 --n-local 30 --min-copy 0.9

The raw objective (glyphgan.evaluate.objective = harmonic mean of precision and coverage) is maximised by copying
the support (identity), so selection is constrained: copy_pix (median NN pixel distance to the support / the
support's own leave-one-out NN distance; ~1.0 for fresh heldout drawings) must be >= `min-copy`.

Phase 1: fixed references (identity, weak, defaults) + `n-random` uniform configs (each transform switched off with
probability `p-off`, so the search also says which components matter). Phase 2: `n-local` multiplicative jitters
of the current constrained best. Resumable: configs already in search.json are skipped.

Outputs (runs/augment/): search.json (all configs + summaries), best.json (best constrained params, its dev summary,
the summaries of weak / identity / defaults / support ceiling, and the objective-vs-copy frontier), best_dev.json
(per-character rows), grid_best.png.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from glyphgan.augment import DEFAULTS, IDENTITY, WEAK, AugmentSampler, save_grid
from glyphgan.data import episodes
from glyphgan.evaluate import evaluate, fmt, objective, save_rows, support_sampler
from glyphgan.features import FeatureExtractor

SPACE = {"rot": (0, 20), "shear": (0, 20), "scale": (0, 0.2), "aniso": (0, 0.2), "trans": (0, 2),
         "alpha": (0, 2), "sigma": (1.5, 6), "thick": (0, 0.8)}
SWITCHABLE = ["rot", "shear", "scale", "aniso", "trans", "alpha", "thick"]
FRONTIER = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def random_config(rng, p_off):
    c = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SPACE.items()}
    for k in SWITCHABLE:
        if rng.random() < p_off:
            c[k] = 0.0
    return c


def jitter(c, rng, s=0.3):
    out = {}
    for k, (lo, hi) in SPACE.items():
        v = c[k] if c[k] > 0 else (0.0 if rng.random() < 0.7 else rng.uniform(lo, hi) * 0.3)
        out[k] = float(np.clip(v * np.exp(rng.normal(0, s)), 0 if k != "sigma" else lo, hi * 1.5))
    return out


def copy_pix(r):
    return r["summary"]["copy_pix"]["mean"]


def best_under(results, min_copy):
    ok = [r for r in results if copy_pix(r) >= min_copy]
    if not ok and min_copy > 0:                                            # nothing feasible: least-copying config
        return max(results, key=copy_pix)
    return max(ok, key=lambda r: r["objective"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/augment")
    ap.add_argument("--clf", default="runs/features/clf.pt")
    ap.add_argument("--n-random", type=int, default=70)
    ap.add_argument("--n-local", type=int, default=30)
    ap.add_argument("--p-off", type=float, default=0.25)
    ap.add_argument("--min-copy", type=float, default=0.9)
    ap.add_argument("--n-draws", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = FeatureExtractor.load(a.clf, dev)
    eps = episodes("dev", max_chars=a.max_chars)
    print(f"{len(eps)} dev episodes, device {dev}", flush=True)

    path = out / "search.json"
    results = json.loads(path.read_text()) if path.exists() else []
    done = {r["name"] for r in results}

    def run(name, params):
        if name in done:
            return
        t0 = time.time()
        _, summ = evaluate(AugmentSampler(params), eps, ext, n_draws=a.n_draws)
        results.append({"name": name, "params": AugmentSampler(params).p, "objective": objective(summ),
                        "summary": summ})
        done.add(name)
        path.write_text(json.dumps(results, indent=1))
        print(f"[{name}] {time.time() - t0:.0f}s {fmt(summ)}", flush=True)

    rng = np.random.default_rng(a.seed)
    randoms = [random_config(rng, a.p_off) for _ in range(a.n_random)]       # drawn upfront: resume-stable
    for name, p in [("identity", IDENTITY), ("weak", WEAK), ("defaults", DEFAULTS)]:
        run(name, p)
    for i, p in enumerate(randoms):
        run(f"rand{i:03d}", p)

    lrng = np.random.default_rng(a.seed + 1)
    for i in range(a.n_local):
        run(f"local{i:03d}", jitter(best_under(results, a.min_copy)["params"], lrng))

    best = best_under(results, a.min_copy)
    rows, summ = evaluate(AugmentSampler(best["params"]), eps, ext, n_draws=a.n_draws)
    save_rows(rows, summ, out / "best_dev.json")
    _, ceil = evaluate(support_sampler, eps, ext, n_draws=a.n_draws)
    by = {r["name"]: r for r in results}
    frontier = []
    for m in FRONTIER:
        b = best_under(results, m)
        if b:
            frontier.append({"min_copy": m, "name": b["name"], "objective": b["objective"], "copy_pix": copy_pix(b),
                             "params": b["params"]})
    (out / "best.json").write_text(json.dumps({
        "selection": f"max objective s.t. copy_pix >= {a.min_copy}",
        "name": best["name"], "params": best["params"], "objective": best["objective"], "summary": summ,
        "weak": by["weak"]["summary"], "identity": by["identity"]["summary"], "defaults": by["defaults"]["summary"],
        "support_ceiling": ceil, "frontier": frontier,
    }, indent=1))
    save_grid(best["params"], out / "grid_best.png")
    print(f"\nBEST {best['name']} {best['params']}\n  {fmt(summ)}")
    for name in ("weak", "identity", "defaults"):
        print(f"{name.upper():9s} {fmt(by[name]['summary'])}")
    print(f"CEILING   {fmt(ceil)}")
    for f in frontier:
        print(f"frontier copy>={f['min_copy']:.1f}: {f['name']} obj={f['objective']:.3f} copy={f['copy_pix']:.2f}")


if __name__ == "__main__":
    main()
