"""Report figures on the fidelity-novelty frontier (README, "Problem and protocol").

    python -m scripts.figures --runs g1_full_lr3e-4 g1_lora4_lr1e-3 "g1_lora4_noaug=LoRA r=4, no DiffAug" \
        --pools g1_lora4_lr1e-3:1000 --out docs/figures [--only frontier,identity,samples]

  frontier.png  x = copy_feat (novelty), y = objective (harmonic mean of precision and coverage), 95% CIs
  identity.png  x = copy_feat, y = identity accuracy
  samples.png   a few dev characters: support | augmentation (runs/augment/best.json) | GAN pool(s)

--runs RUN[=FAMILY]: runs under <root>/adapt/; the family (colour + line style) defaults to "Full FT" / "LoRA r=<r>"
(+ "no DiffAug" when shift = 0). Each run is a line through its training steps; eval labels "<step>_<variant>"
(t<psi> = truncation, r<n>x<eta> = critic-guided refinement) are drawn as separate markers linked to their step.

CI of the objective: the objective is a function of two means, so its 95% CI uses the delta method. With per-character
rows (adapt evals) the linearised per-character values a*P_i + b*C_i give the exact delta-method interval; without rows
(augmentation search, summaries only) a*ci_P + b*ci_C is used, an upper bound (correlation 1 between P and C).
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

# Categorical palette (colour-blind-checked order), greys for the baseline, text inks.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (8, 2)), (0, (1, 1)), (0, (3, 1, 3, 3))]
GREY_PT, GREY_LINE, INK, INK2 = "#b4b2ad", "#52514e", "#0b0b0b", "#52514e"
SHADE = "#efeeea"
VARIANT_MARKERS = {"t": "v", "r": "D"}
EXTRA_MARKERS = ["s", "P", "X", "*", "^", "h"]
COPY_MIN = 0.9
XMAX = 1.4          # default right edge: early steps (<= 100) and strong augmentations sit beyond, far from the rule

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK, "axes.edgecolor": "#8a8984", "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})


# ---------------------------------------------------------------- data
def harmonic(p, c):
    return 2 * p * c / (p + c) if p + c > 0 else 0.0


def objective_ci(summary: dict, rows: list[dict] | None) -> float:
    p, c = summary["precision"]["mean"], summary["coverage"]["mean"]
    a, b = 2 * c ** 2 / (p + c) ** 2, 2 * p ** 2 / (p + c) ** 2           # dH/dP, dH/dC
    if rows:
        h = np.array([a * r["precision"] + b * r["coverage"] for r in rows])
        return float(stats.t.ppf(0.975, len(h) - 1) * h.std(ddof=1) / np.sqrt(len(h)))
    return a * summary["precision"]["ci"] + b * summary["coverage"]["ci"]


def point(summary: dict, rows: list[dict] | None = None) -> dict:
    """Plot coordinates of one evaluation: means and 95% CI half-widths."""
    s = summary
    return {"copy": s["copy_feat"]["mean"], "copy_ci": s["copy_feat"]["ci"],
            "objective": harmonic(s["precision"]["mean"], s["coverage"]["mean"]),
            "objective_ci": objective_ci(s, rows),
            "identity": s["identity"]["mean"], "identity_ci": s["identity"]["ci"], "n": s["n_chars"]}


def default_family(cfg: dict) -> str:
    fam = "Full FT" if cfg.get("mode") == "full" else f"LoRA r={cfg.get('r')}"
    return fam + (", no DiffAug" if cfg.get("shift", 1) == 0 else "")


def variant_name(v: str) -> str:
    if m := re.fullmatch(r"t([\d.]+)", v):
        return f"truncation ψ={m[1]}"
    if m := re.fullmatch(r"r(\d+)x([\d.]+)", v):
        return f"refinement {m[1]}×{m[2]}"
    return v


def load_run(root: Path, spec: str, min_step: int) -> dict:
    name, _, fam = spec.partition("=")
    d = root / "adapt" / name
    cfg = json.loads((d / "config.json").read_text())
    steps, variants = [], []
    for f in d.glob("eval_*.json"):
        label = f.stem[len("eval_"):]
        step, _, var = label.partition("_")
        if int(step) < min_step:
            continue
        e = json.loads(f.read_text())
        pt = point(e["summary"], e.get("rows")) | {"step": int(step), "variant": var}
        (variants if var else steps).append(pt)
    assert steps or variants, f"no eval_*.json with step >= {min_step} in {d}"
    return {"name": name, "family": fam or default_family(cfg), "steps": sorted(steps, key=lambda p: p["step"]),
            "variants": sorted(variants, key=lambda p: (p["step"], p["variant"]))}


def load_augment(path: Path) -> list[dict]:
    return [point(r["summary"]) | {"name": r["name"]} for r in json.loads(path.read_text())]


def pareto_upper(pts: list[dict], y: str) -> list[dict]:
    """Points not dominated in (copy_feat up, y up), sorted by decreasing copy_feat."""
    out, best = [], -np.inf
    for p in sorted(pts, key=lambda p: (-p["copy"], -p[y])):
        if p[y] > best:
            out.append(p)
            best = p[y]
    return out


# ---------------------------------------------------------------- scatter figures
def frontier_fig(runs, aug, real, y: str, ylabel: str, title: str, out: Path, annotate: str | None,
                 xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ycol = y

    # reference regions / lines (drawn first, behind data)
    ax.axvspan(-1, COPY_MIN, color=SHADE, zorder=0, lw=0)
    ax.axvline(1.0, color=INK2, ls="--", lw=1, zorder=1)
    ax.axhline(real[ycol], color=INK2, ls=":", lw=1.2, zorder=1)

    # augmentation baseline: cloud + upper Pareto frontier (staircase = best y achievable at copy_feat >= x)
    ax.scatter([p["copy"] for p in aug], [p[ycol] for p in aug], s=14, color=GREY_PT, lw=0, zorder=2,
               label=f"Augmentation configs (n={len(aug)})")
    fr = pareto_upper(aug, ycol)
    xs = [p["copy"] for p in fr]
    ax.step(xs, [p[ycol] for p in fr], where="post", color=GREY_LINE, lw=1.6, zorder=3,
            label="Augmentation frontier")
    ax.errorbar(xs, [p[ycol] for p in fr], xerr=[p["copy_ci"] for p in fr], yerr=[p[ycol + "_ci"] for p in fr],
                fmt="o", ms=4, color=GREY_LINE, ecolor=GREY_LINE, elinewidth=0.8, capsize=0, zorder=3)

    # GAN runs: one colour + line style per family
    fams = list(dict.fromkeys(r["family"] for r in runs))
    style = {f: (PALETTE[i % len(PALETTE)], LINESTYLES[i % len(LINESTYLES)]) for i, f in enumerate(fams)}
    vmark: dict[str, str] = {}
    seen = set()
    for r in runs:
        col, ls = style[r["family"]]
        st = r["steps"]
        lab = r["family"] if r["family"] not in seen else None
        seen.add(r["family"])
        if st:
            ax.errorbar([p["copy"] for p in st], [p[ycol] for p in st], xerr=[p["copy_ci"] for p in st],
                        yerr=[p[ycol + "_ci"] for p in st], color=col, ls=ls, lw=2, marker="o", ms=5,
                        mec="white", mew=0.8, elinewidth=0.8, ecolor=col, capsize=0, zorder=5, label=lab)
            if r["name"] == annotate:
                for p in st:
                    ax.annotate(f"{p['step']}", (p["copy"], p[ycol]), xytext=(-4, 6), textcoords="offset points",
                                fontsize=8, color=INK, ha="right", zorder=7,
                                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
        by_step = {p["step"]: p for p in st}
        for v in r["variants"]:
            kind = v["variant"][0]
            if v["variant"] not in vmark:
                m = VARIANT_MARKERS.get(kind)
                if m is None or m in vmark.values():
                    m = next((e for e in EXTRA_MARKERS if e not in vmark.values()), "o")
                vmark[v["variant"]] = m
            if (b := by_step.get(v["step"])) is not None:
                ax.plot([b["copy"], v["copy"]], [b[ycol], v[ycol]], color=col, ls=":", lw=1, zorder=4)
            ax.errorbar(v["copy"], v[ycol], xerr=v["copy_ci"], yerr=v[ycol + "_ci"], fmt=vmark[v["variant"]],
                        ms=7, color=col, mec=INK, mew=0.7, elinewidth=0.8, ecolor=col, capsize=0, zorder=6)
    for var, m in vmark.items():                                           # legend-only handles for variants
        ax.plot([], [], m, ms=7, color="white", mec=INK, mew=0.9, ls="none", label=variant_name(var))

    # axes, labels on the reference marks
    ax.set_xlabel("Novelty: copy ratio in feature space (copy_feat; 1 = fresh real data)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    gan = [p for r in runs for p in r["steps"] + r["variants"]]
    x0, x1 = xlim or (-0.03, XMAX)
    vis = [p for p in aug + gan if x0 <= p["copy"] <= x1]
    ally = [p[ycol] for p in vis] + [real[ycol]]
    y0, y1 = ylim or (min(ally) - 0.03, max(ally) + 0.03)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.text(x0 + 0.01 * (x1 - x0), y0 + 0.02 * (y1 - y0), f"copy-like (copy_feat < {COPY_MIN})", color=INK2,
            fontsize=8.5, va="bottom")
    ax.text(1.0, y0 + 0.02 * (y1 - y0), " fresh real data", color=INK2, fontsize=8.5, va="bottom", ha="left")
    ax.text(x0 + 0.01 * (x1 - x0), real[ycol], f"real support (ceiling) {real[ycol]:.3f}", color=INK2,
            fontsize=8.5, va="bottom", ha="left")

    ns = sorted({p["n"] for p in gan})
    hidden = [p for p in gan if not (x0 <= p["copy"] <= x1 and y0 <= p[ycol] <= y1)]
    hid_aug = sum(not (x0 <= p["copy"] <= x1 and y0 <= p[ycol] <= y1) for p in aug)
    note = (f"Dev split. GAN runs: {'/'.join(map(str, ns))} characters; augmentation and real reference: "
            f"{aug[0]['n']} characters. Error bars: 95% CI across characters (objective: delta method).")
    if hidden or hid_aug:
        steps = sorted({p["step"] for p in hidden})
        note += (f"\nOutside the plotted range: {len(hidden)} GAN points (steps {', '.join(map(str, steps))})"
                 f" and {hid_aug} augmentation configs." if hidden else
                 f"\nOutside the plotted range: {hid_aug} augmentation configs.")
    ax.text(0.0, -0.13, note, fontsize=8, color=INK2, ha="left", va="top", transform=ax.transAxes)

    # legend order: baseline, method families, sampling variants
    h, lab = ax.get_legend_handles_labels()
    order = {k: i for i, k in enumerate([f"Augmentation configs (n={len(aug)})", "Augmentation frontier"] + fams)}
    idx = sorted(range(len(lab)), key=lambda i: order.get(lab[i], len(order)))
    ax.legend([h[i] for i in idx], [lab[i] for i in idx], loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9, handlelength=2.6)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------- samples
def strip(imgs: np.ndarray, sep: int = 2) -> np.ndarray:
    """(n, 784) uint8 -> one row image, ink dark on white, separated by light gaps."""
    n = len(imgs)
    out = np.full((28, n * 28 + (n - 1) * sep), 235, dtype=np.uint8)
    for i, im in enumerate(imgs):
        out[:, i * (28 + sep): i * (28 + sep) + 28] = 255 - im.reshape(28, 28)
    return out


def samples_fig(root: Path, pools: list[str], best_path: Path, n_chars: int, char_ids: list[int] | None, n: int,
                seed: int, out: Path):
    import torch

    from glyphgan.augment import AugmentSampler
    from glyphgan.data import episodes

    loaded = []
    for spec in pools:
        spec, _, title = spec.partition("=")
        run, _, label = spec.partition(":")
        d = torch.load(root / "adapt" / run / f"pools_{label}.pt", weights_only=False)
        ids = [int(c) for c in d["char_ids"]]
        loaded.append((title or f"GAN: {run} @ {label}",{c: i for i, c in enumerate(ids)}, d["pool"]))
    common = set.intersection(*(set(m) for _, m, _ in loaded))
    eps = {ep.char: ep for ep in episodes("dev") if ep.char in common}
    if char_ids is None:
        rng = np.random.default_rng(seed)
        char_ids = sorted(rng.choice(sorted(eps), size=min(n_chars, len(eps)), replace=False).tolist())
    best = json.loads(best_path.read_text())
    aug = AugmentSampler(best["params"])

    cols = ["Support (10 real)", "Augmentation baseline"] + [t for t, _, _ in loaded]
    fig, axes = plt.subplots(len(char_ids), len(cols), figsize=(3.6 * len(cols), 0.55 * len(char_ids) + 0.5),
                             squeeze=False, gridspec_kw={"wspace": 0.08, "hspace": 0.12})
    for i, c in enumerate(char_ids):
        ep = eps[c]
        g = torch.Generator().manual_seed(seed * 1_000_003 + c)
        u8 = lambda x: ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()  # noqa: E731
        blocks = [u8(ep.support[:n]), u8(aug(ep, n, g))] + [pool[m[c], :n].numpy() for _, m, pool in loaded]
        for j, b in enumerate(blocks):
            ax = axes[i, j]
            ax.imshow(strip(b), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(cols[j], fontsize=10, fontweight="bold", loc="left")
            if j == 0:
                ax.set_ylabel(ep.name.replace("/character", "\n#"), rotation=0, ha="right", va="center",
                              fontsize=8, color=INK2)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out} (chars {char_ids})")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="RUN[=FAMILY] under <root>/adapt")
    ap.add_argument("--pools", nargs="*", default=[], help="RUN:LABEL[=TITLE] -> <root>/adapt/RUN/pools_LABEL.pt")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--search", default="runs/augment/search.json")
    ap.add_argument("--best", default="runs/augment/best.json")
    ap.add_argument("--real", default="runs/eval/support_dev.json")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--only", default="frontier,identity,samples")
    ap.add_argument("--min-step", type=int, default=1, help="drop steps below (0 = unadapted generator)")
    ap.add_argument("--annotate", default=None, help="run whose points get step labels (default: first run)")
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="frontier.png y range")
    ap.add_argument("--ylim-identity", type=float, nargs=2, default=None)
    ap.add_argument("--n-chars", type=int, default=5)
    ap.add_argument("--chars", type=int, nargs="*", default=None, help="explicit character ids for samples.png")
    ap.add_argument("--n", type=int, default=10, help="images per block in samples.png")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(","))
    if only & {"frontier", "identity"}:
        runs = [load_run(root, s, a.min_step) for s in a.runs]
        aug = load_augment(Path(a.search))
        real_e = json.loads(Path(a.real).read_text())
        real = point(real_e["summary"], real_e.get("rows"))
        ann = a.annotate or (runs[0]["name"] if runs else None)
        if "frontier" in only:
            frontier_fig(runs, aug, real, "objective", "Fidelity: objective = harmonic mean(precision, coverage)",
                         "GAN vs augmentation on the fidelity–novelty plane", out / "frontier.png", ann,
                         a.xlim, a.ylim)
        if "identity" in only:
            frontier_fig(runs, aug, real, "identity", "Identity accuracy (nearest prototype in the alphabet)",
                         "Do samples stay recognisable as the target character?", out / "identity.png", ann,
                         a.xlim, a.ylim_identity)
    if "samples" in only:
        assert a.pools, "samples.png needs --pools RUN:LABEL"
        samples_fig(root, a.pools, Path(a.best), a.n_chars, a.chars, a.n, a.seed, out / "samples.png")


if __name__ == "__main__":
    main()
