"""Compact table of evaluation summaries.  python -m scripts.table runs/adapt/g1_* [--steps 1000,2000] [--json f.json ...]"""
import argparse
import json
from pathlib import Path

COLS = ["precision", "recall", "density", "coverage", "copy_pix", "copy_feat", "identity", "critic_gap"]


def row(name, s):
    cells = [f"{s[c]['mean']:.3f}" if c in s else "  -  " for c in COLS]
    return f"{name:34s} " + " ".join(f"{c:>9s}" for c in cells) + f" {s.get('objective', float('nan')):9.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*")
    p.add_argument("--steps", default=None)
    p.add_argument("--json", nargs="*", default=[])
    p.add_argument("--prefix", default="eval")
    a = p.parse_args()
    print(f"{'run@step':34s} " + " ".join(f"{c[:9]:>9s}" for c in COLS) + f" {'objective':>9s}")
    for f in a.json:
        print(row(Path(f).stem, json.loads(Path(f).read_text())["summary"]))
    keep = None if a.steps is None else {int(s) for s in a.steps.split(",")}
    for r in map(Path, a.runs):
        for f in sorted(r.glob(f"{a.prefix}_*.json"), key=lambda q: (int(q.stem.split("_")[1]), q.stem)):
            label = f.stem[len(a.prefix) + 1:]
            if keep is None or int(label.split("_")[0]) in keep:
                print(row(f"{r.name}@{label}", json.loads(f.read_text())["summary"]))


if __name__ == "__main__":
    main()
