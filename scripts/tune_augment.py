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


def copy_feat(r):
    return r["summary"]["copy_feat"]["mean"]


def pareto(results):
    """Upper frontier of (copy_feat, objective): configs not beaten on both axes."""
    out, best = [], -1.0
    for r in sorted(results, key=lambda r: (-copy_feat(r), -r["objective"])):
        if r["objective"] > best:
            out.append(r)
            best = r["objective"]
    return out


def to_u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def replay(a, ext):
    """Pre-registered rule (README, "Problem and protocol"): max dev objective s.t. copy_feat >= 0.9, replayed once on test, plus
    the dev Pareto frontier, references and the real ceiling on test; uint8 pools for scripts.eval_pools."""
    out = Path(a.out)
    results = json.loads((out / "search.json").read_text())
    feasible = [r for r in results if copy_feat(r) >= a.min_copy_feat]
    sel = max(feasible, key=lambda r: r["objective"])
    (out / "selected_feat09.json").write_text(json.dumps({
        "selection": f"max dev objective s.t. copy_feat >= {a.min_copy_feat}", "name": sel["name"],
        "params": sel["params"], "objective": sel["objective"], "summary": sel["summary"]}, indent=1))
    print(f"selected {sel['name']} {sel['params']}\n  dev {fmt(sel['summary'])}", flush=True)

    best_name = json.loads((out / "best.json").read_text())["name"]
    by = {r["name"]: r for r in results}
    tags: dict[str, list[str]] = {}
    for name, tag in [(sel["name"], "selected_feat09"), (best_name, "best_copy_pix"), ("weak", "weak"),
                      ("identity", "identity")] + [(r["name"], "dev_frontier") for r in pareto(results)]:
        tags.setdefault(name, []).append(tag)

    test = episodes("test")
    print(f"{len(test)} test episodes", flush=True)
    rows_out = []
    for name, tg in tags.items():
        _, summ = evaluate(AugmentSampler(by[name]["params"]), test, ext, n_draws=a.n_draws)
        rows_out.append({"name": name, "params": by[name]["params"], "objective": objective(summ), "summary": summ,
                         "tags": tg, "dev_objective": by[name]["objective"], "dev_copy_feat": copy_feat(by[name])})
        (out / "test_frontier.json").write_text(json.dumps(rows_out, indent=1))
        print(f"[test {name} {','.join(tg)}] {fmt(summ)}", flush=True)
    rows, summ = evaluate(support_sampler, test, ext, n_draws=a.n_draws)
    save_rows(rows, summ, "runs/eval/support_test.json")
    print(f"[test support ceiling] {fmt(summ)}", flush=True)

    for tag, name in [("aug_feat09", sel["name"]), ("aug_best", best_name)]:
        dump_pools(a, tag, name, by[name]["params"], test)


def dump_pools(a, tag, name, params, test):
    sampler = AugmentSampler(params)
    for part, eps in [("dev", episodes("dev")), ("test", test)]:
        d = Path("runs/adapt") / f"{tag}_{part}"
        d.mkdir(parents=True, exist_ok=True)
        pool = torch.stack([to_u8(sampler(ep, a.pool, torch.Generator().manual_seed(a.seed * 1_000_003 + ep.char)))
                            for ep in eps])
        torch.save({"char_ids": torch.tensor([ep.char for ep in eps]), "pool": pool}, d / "pools_0.pt")
        (d / "config.json").write_text(json.dumps({"part": part, "max_chars": None, "method": "augment",
                                                   "name": name, "params": params,
                                                   "pool": a.pool, "seed": a.seed}, indent=1))
        print(f"pool {d} {tuple(pool.shape)}", flush=True)


def search_v2(a, ext):
    """Local search under the actual rule (copy_feat >= min_copy_feat), seeded by the v1 dev frontier points with
    copy_feat in [0.8, 1.0] + interpolations between the selected v1 config and the best points just below 0.9.
    New points -> search_v2.json; selection over v1 + v2 -> selected_feat09_v2.json (with its test replay); pools."""
    out = Path(a.out)
    v1 = json.loads((out / "search.json").read_text())
    path = out / "search_v2.json"
    v2 = json.loads(path.read_text()) if path.exists() else []
    done = {r["name"] for r in v2}
    eps = episodes("dev")

    def run(name, params):
        if name in done:
            return
        t0 = time.time()
        _, summ = evaluate(AugmentSampler(params), eps, ext, n_draws=a.n_draws)
        v2.append({"name": name, "params": AugmentSampler(params).p, "objective": objective(summ), "summary": summ})
        done.add(name)
        path.write_text(json.dumps(v2, indent=1))
        print(f"[{name}] {time.time() - t0:.0f}s cf={copy_feat(v2[-1]):.3f} obj={v2[-1]['objective']:.3f} "
              f"{fmt(summ)}", flush=True)

    def best_feasible():
        ok = [r for r in v1 + v2 if copy_feat(r) >= a.min_copy_feat]
        return max(ok, key=lambda r: r["objective"])

    front = [r for r in pareto(v1) if 0.8 <= copy_feat(r) <= 1.0]
    print("seeds:", [(r["name"], round(copy_feat(r), 3), round(r["objective"], 3)) for r in front], flush=True)
    sel1 = best_feasible()
    below = sorted([r for r in front if copy_feat(r) < a.min_copy_feat], key=lambda r: -r["objective"])[:2]
    for b in below:                                                       # interpolations sel1 <-> points below 0.9
        for t in (0.2, 0.35, 0.5, 0.65, 0.8):
            run(f"v2_interp_{b['name']}_{int(t * 100)}",
                {k: (1 - t) * sel1["params"][k] + t * b["params"][k] for k in SPACE})
    rng = np.random.default_rng(a.seed + 2)
    for i in range(a.n_seed_jitter):                                      # perturbations around the frontier seeds
        run(f"v2_seed{i:03d}", jitter(front[i % len(front)]["params"], rng, s=0.25))
    for i in range(a.n_local):                                            # adaptive: around the current best feasible
        run(f"v2_local{i:03d}", jitter(best_feasible()["params"], rng, s=0.2 if i % 2 else 0.35))

    sel = best_feasible()
    test = episodes("test")
    _, tsum = evaluate(AugmentSampler(sel["params"]), test, ext, n_draws=a.n_draws)
    (out / "selected_feat09_v2.json").write_text(json.dumps({
        "selection": f"max dev objective s.t. copy_feat >= {a.min_copy_feat}, over search.json + search_v2.json",
        "name": sel["name"], "params": sel["params"], "objective": sel["objective"], "summary": sel["summary"],
        "test_objective": objective(tsum), "test_summary": tsum}, indent=1))
    print(f"\nSELECTED v2 {sel['name']} {sel['params']}\n  dev  {fmt(sel['summary'])}\n  test {fmt(tsum)}", flush=True)
    dump_pools(a, "aug_feat09v2", sel["name"], sel["params"], test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="apply the copy_feat rule, evaluate on test, dump pools")
    ap.add_argument("--search-v2", action="store_true", help="local search under the copy_feat rule (+ test, pools)")
    ap.add_argument("--n-seed-jitter", type=int, default=30)
    ap.add_argument("--min-copy-feat", type=float, default=0.9)
    ap.add_argument("--pool", type=int, default=100)
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
    if a.replay:
        return replay(a, ext)
    if a.search_v2:
        return search_v2(a, ext)
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
