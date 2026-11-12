"""Figures for the README: fidelity-novelty frontier, identity, sample grids.

    python -m scripts.figures --runs g1_full_lr3e-4 g1_lora4_lr1e-3 "g1_lora4_noaug=LoRA r=4, no DiffAug" \
        --pools g1_lora4_lr1e-3:1000 --out docs/figures [--only frontier,identity,samples]

  frontier.png  x = copy_feat (novelty), y = objective (harmonic mean of precision and coverage), 95% CIs
  identity.png  x = copy_feat, y = identity accuracy
  samples.png   a few characters of --part: support | pools given by --pools (augmentation pools included)
                [| on-the-fly AugmentSampler column with --aug-params]

--selected RUN:LABEL and --aug-selected mark the configurations chosen by the selection rule (stars).
Each run is drawn as a line through its eval steps; variants (t = truncation, r = refinement) as extra markers.
The objective CI uses the delta method (harmonic mean of two means).
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


def load_run(root: Path, spec: str, min_step: int, keep: set[str] | None = None) -> dict:
    """keep: label suffixes to load ("base" = plain training steps); None = all."""
    name, _, fam = spec.partition("=")
    d = root / "adapt" / name
    cfg = json.loads((d / "config.json").read_text())
    steps, variants = [], []
    for f in d.glob("eval_*.json"):
        label = f.stem[len("eval_"):]
        step, _, var = label.partition("_")
        if int(step) < min_step or (keep is not None and (var or "base") not in keep):
            continue
        e = json.loads(f.read_text())
        pt = point(e["summary"], e.get("rows")) | {"step": int(step), "variant": var}
        (variants if var else steps).append(pt)
    assert steps or variants, f"no eval_*.json with step >= {min_step} in {d}"
    return {"name": name, "family": fam or default_family(cfg), "steps": sorted(steps, key=lambda p: p["step"]),
            "variants": sorted(variants, key=lambda p: (p["step"], p["variant"]))}


def _search_entries(paths: list[Path]) -> list[dict]:
    """Concatenation of several search lists (e.g. search.json + search_v2.json); later files win on name clashes."""
    by = {}
    for p in paths:
        for r in json.loads(Path(p).read_text()):
            by[r["name"]] = r
    return list(by.values())


def load_augment(paths: list[Path]) -> list[dict]:
    return [point(r["summary"]) | {"name": r["name"]} for r in _search_entries(paths)]


def load_aug_selected(spec: str | None, search: list[Path]) -> dict | None:
    """The rule-selected augmentation config: a JSON file with a "summary" (e.g. runs/augment/selected_feat09.json),
    or a name / tag looked up in the --search list (e.g. "selected_feat09" in runs/augment/test_frontier.json)."""
    if not spec:
        return None
    if Path(spec).is_file():
        e = json.loads(Path(spec).read_text())
        return point(e["summary"], e.get("rows")) | {"name": e.get("name", Path(spec).stem)}
    hits = [e for e in _search_entries(search) if e["name"] == spec or spec in e.get("tags", [])]
    assert len(hits) == 1, f"--aug-selected {spec}: {len(hits)} matches in {search}"
    return point(hits[0]["summary"]) | {"name": hits[0]["name"]}


def load_selected(root: Path, spec: str | None) -> dict | None:
    """RUN:LABEL -> the GAN point selected by the rule (<root>/adapt/RUN/eval_LABEL.json)."""
    if not spec:
        return None
    run, _, label = spec.partition(":")
    e = json.loads((root / "adapt" / run / f"eval_{label}.json").read_text())
    return point(e["summary"], e.get("rows")) | {"name": f"{run}@{label}"}


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
                 xlim=None, ylim=None, sel_gan: dict | None = None, sel_aug: dict | None = None, part: str = "dev"):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ycol = y

    # axis limits first, so that out-of-range GAN points can be dropped (a line must not run off to a clipped point)
    x0, x1 = xlim or (-0.03, XMAX)
    gan_all = [p for r in runs for p in r["steps"] + r["variants"]]
    extra = [p for p in (sel_gan, sel_aug) if p]
    ally = [p[ycol] for p in aug + gan_all + extra if x0 <= p["copy"] <= x1] + [real[ycol]]
    y0, y1 = ylim or (min(ally) - 0.03, max(ally) + 0.03)
    inside = lambda p: x0 <= p["copy"] <= x1 and y0 <= p[ycol] <= y1  # noqa: E731
    hidden = [(r["name"], p) for r in runs for p in r["steps"] + r["variants"] if not inside(p)]

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

    # GAN runs: one colour + line style per family; only in-range points
    fams = list(dict.fromkeys(r["family"] for r in runs))
    style = {f: (PALETTE[i % len(PALETTE)], LINESTYLES[i % len(LINESTYLES)]) for i, f in enumerate(fams)}
    vmark: dict[str, str] = {}
    seen = set()
    for r in runs:
        col, ls = style[r["family"]]
        st = [p for p in r["steps"] if inside(p)]
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
        elif lab:
            ax.plot([], [], color=col, ls=ls, lw=2, marker="o", ms=5, mec="white", label=lab)
        by_step = {p["step"]: p for p in st}
        for v in r["variants"]:
            if not inside(v):
                continue
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

    # configurations selected by the rule (max dev objective s.t. copy_feat >= 0.9): stars + short labels
    for p, txt, col, off, ha in [(sel_gan, "selected GAN", INK, (8, 8), "left"),
                                 (sel_aug, "selected augmentation", GREY_LINE, (-10, 9), "right")]:
        if p and inside(p):
            ax.errorbar(p["copy"], p[ycol], xerr=p["copy_ci"], yerr=p[ycol + "_ci"], fmt="*", ms=17, color="#ffd23f"
                        if col == INK else "white", mec=col, mew=1.3, ecolor=col, elinewidth=1, capsize=0, zorder=8)
            ax.annotate(f"{txt} ({p[ycol]:.3f})", (p["copy"], p[ycol]), xytext=off, textcoords="offset points",
                        fontsize=8.5, fontweight="bold", color=col, zorder=9, ha=ha,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    if sel_gan or sel_aug:
        ax.plot([], [], "*", ms=13, color="white", mec=INK, mew=1.1, ls="none",
                label=f"Selected by the rule (copy_feat ≥ {COPY_MIN})")

    # axes, labels on the reference marks
    ax.set_xlabel("Novelty: copy ratio in feature space (copy_feat; 1 = fresh real data)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.text(x0 + 0.01 * (x1 - x0), y0 + 0.02 * (y1 - y0), f"copy-like (copy_feat < {COPY_MIN})", color=INK2,
            fontsize=8.5, va="bottom")
    ax.text(1.0, y0 + 0.02 * (y1 - y0), " fresh real data", color=INK2, fontsize=8.5, va="bottom", ha="left")
    ax.text(x0 + 0.01 * (x1 - x0), real[ycol], f"real support (ceiling) {real[ycol]:.3f}", color=INK2,
            fontsize=8.5, va="bottom", ha="left")

    ns = sorted({p["n"] for p in gan_all})
    hid_aug = sum(not inside(p) for p in aug)
    note = (f"{part.capitalize()} split. GAN runs: {'/'.join(map(str, ns))} characters; augmentation and real "
            f"reference: {aug[0]['n']} characters. Error bars: 95% CI across characters (objective: delta method).")
    parts = []
    if hidden:
        lbl = sorted({f"{p['step']}" + (f"_{p['variant']}" if p["variant"] else "") for _, p in hidden},
                     key=lambda s: (int(s.split("_")[0]), s))
        parts.append(f"{len(hidden)} GAN points ({', '.join(lbl)}) not drawn")
    if hid_aug:
        parts.append(f"{hid_aug} augmentation configs clipped")
    if parts:
        note += f"\nOutside the plotted range: {'; '.join(parts)}."
    ax.text(0.0, -0.13, note, fontsize=8, color=INK2, ha="left", va="top", transform=ax.transAxes)

    # legend order: baseline, method families, sampling variants, selection
    h, lab = ax.get_legend_handles_labels()
    order = {k: i for i, k in enumerate([f"Augmentation configs (n={len(aug)})", "Augmentation frontier"] + fams)}
    idx = sorted(range(len(lab)), key=lambda i: order.get(lab[i], len(order)))
    ax.legend([h[i] for i in idx], [lab[i] for i in idx], loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9, handlelength=2.6)
    fig.savefig(out)
    plt.close(fig)
    gone = ["%s@%d%s" % (n, p["step"], "_" + p["variant"] if p["variant"] else "") for n, p in hidden]
    print(f"wrote {out}" + (f" (not drawn: {gone})" if gone else ""))


# ---------------------------------------------------------------- samples
def strip(imgs: np.ndarray, sep: int = 2) -> np.ndarray:
    """(n, 784) uint8 -> one row image, ink dark on white, separated by light gaps."""
    n = len(imgs)
    out = np.full((28, n * 28 + (n - 1) * sep), 235, dtype=np.uint8)
    for i, im in enumerate(imgs):
        out[:, i * (28 + sep): i * (28 + sep) + 28] = 255 - im.reshape(28, 28)
    return out


def samples_fig(root: Path, pools: list[str], best_path: Path | None, n_chars: int, char_ids: list[int] | None, n: int,
                seed: int, out: Path, part: str = "dev"):
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
    eps = {ep.char: ep for ep in episodes(part) if ep.char in common}
    if char_ids is None:
        rng = np.random.default_rng(seed)
        char_ids = sorted(rng.choice(sorted(eps), size=min(n_chars, len(eps)), replace=False).tolist())
    aug = AugmentSampler(json.loads(best_path.read_text())["params"]) if best_path else None

    cols = ["Support (10 real)"] + (["Augmentation (on the fly)"] if aug else []) + [t for t, _, _ in loaded]
    fig, axes = plt.subplots(len(char_ids), len(cols), figsize=(3.6 * len(cols), 0.55 * len(char_ids) + 0.5),
                             squeeze=False, gridspec_kw={"wspace": 0.16, "hspace": 0.12})
    for i, c in enumerate(char_ids):
        ep = eps[c]
        g = torch.Generator().manual_seed(seed * 1_000_003 + c)
        u8 = lambda x: ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()  # noqa: E731
        blocks = [u8(ep.support[:n])] + ([u8(aug(ep, n, g))] if aug else []) + \
            [pool[m[c], :n].numpy() for _, m, pool in loaded]
        for j, b in enumerate(blocks):
            ax = axes[i, j]
            ax.imshow(strip(b), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(cols[j], fontsize=9.5, fontweight="bold", loc="left")
            if j == 0:
                ax.set_ylabel(ep.name.replace("_", " ").replace(" (", "\n(").replace("/character", "\n#"), rotation=0, ha="right", va="center",
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
    ap.add_argument("--search", nargs="+", default=["runs/augment/search.json"],
                    help="augmentation search list(s), merged (e.g. search.json search_v2.json)")
    ap.add_argument("--aug-params", default=None,
                    help="samples.png: extra on-the-fly AugmentSampler column from this JSON's 'params' (e.g. "
                         "runs/augment/selected_feat09.json); default none, pass augmentation pools via --pools")
    ap.add_argument("--variants", default=None,
                    help="label suffixes to show, comma list: 'base' = training steps, e.g. base,r10x0.1 (default all)")
    ap.add_argument("--selected", default=None, help="RUN:LABEL of the selected GAN configuration (star)")
    ap.add_argument("--aug-selected", default=None,
                    help="selected augmentation config (star): a JSON with 'summary' or a name/tag in --search")
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
    ap.add_argument("--part", default="dev", help="dev or test (footnote and episodes for samples.png)")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(","))
    if only & {"frontier", "identity"}:
        keep = set(a.variants.split(",")) if a.variants else None
        runs = [load_run(root, s, a.min_step, keep) for s in a.runs]
        aug = load_augment([Path(s) for s in a.search])
        real_e = json.loads(Path(a.real).read_text())
        real = point(real_e["summary"], real_e.get("rows"))
        sel = dict(sel_gan=load_selected(root, a.selected), sel_aug=load_aug_selected(a.aug_selected,
                                                                                  [Path(s) for s in a.search]))
        ann = a.annotate or (runs[0]["name"] if runs else None)
        if "frontier" in only:
            frontier_fig(runs, aug, real, "objective", "Fidelity: objective = harmonic mean(precision, coverage)",
                         "GAN vs augmentation on the fidelity–novelty plane", out / "frontier.png", ann,
                         a.xlim, a.ylim, **sel, part=a.part)
        if "identity" in only:
            frontier_fig(runs, aug, real, "identity", "Identity accuracy (nearest prototype in the alphabet)",
                         "Do samples stay recognisable as the target character?", out / "identity.png", ann,
                         a.xlim, a.ylim_identity, **sel, part=a.part)
    if "samples" in only:
        assert a.pools, "samples.png needs --pools RUN:LABEL"
        samples_fig(root, a.pools, Path(a.aug_params) if a.aug_params else None, a.n_chars, a.chars, a.n, a.seed,
                    out / "samples.png", a.part)


if __name__ == "__main__":
    main()
