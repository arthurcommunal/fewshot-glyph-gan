"""Pre-registered selection rule (README, "Problem and protocol"): on dev, maximise the objective subject to copy_feat >= 0.9.

    python -m scripts.select_rule runs/adapt/g2_* [--min_copy 0.9] [--out runs/select_dev.json]

Prints the best (run, step/variant) per run and overall; candidates are every eval_<label>.json of each run.
"""
import argparse
import json
from pathlib import Path


def candidates(run: Path):
    for f in run.glob("eval_*.json"):
        s = json.loads(f.read_text())["summary"]
        yield {"run": run.name, "label": f.stem[5:], "objective": s["objective"], "copy_feat": s["copy_feat"]["mean"],
               "precision": s["precision"]["mean"], "coverage": s["coverage"]["mean"], "identity": s["identity"]["mean"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--min_copy", type=float, default=0.9)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    best = {}
    for run in map(Path, a.runs):
        ok = [c for c in candidates(run) if c["copy_feat"] >= a.min_copy]
        if ok:
            best[run.name] = max(ok, key=lambda c: c["objective"])
    for b in sorted(best.values(), key=lambda c: -c["objective"]):
        print(f"{b['run']:24s} {b['label']:>16s} objective={b['objective']:.3f} copy_feat={b['copy_feat']:.3f} "
              f"P={b['precision']:.3f} C={b['coverage']:.3f} id={b['identity']:.3f}")
    if best:
        overall = max(best.values(), key=lambda c: c["objective"])
        print("OVERALL", json.dumps(overall))
        if a.out:
            Path(a.out).write_text(json.dumps({"rule": f"max objective s.t. copy_feat >= {a.min_copy}",
                                               "per_run": best, "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
