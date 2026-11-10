"""Metrics for every pools_<step>.pt of adaptation runs (+ critic-gap diagnostics from diag.jsonl).

    python -m scripts.eval_pools runs/adapt/lora_r4 [runs/adapt/full ...]

Writes <run>/eval_<step>.json (rows + summary) and prints one line per (run, step).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from glyphgan.data import episodes
from glyphgan.evaluate import evaluate_pools, fmt, save_rows
from glyphgan.features import FeatureExtractor


def main():
    ext = FeatureExtractor.load("runs/features/clf.pt", "cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    for run in map(Path, sys.argv[1:]):
        cfg = json.loads((run / "config.json").read_text())
        key = (cfg.get("part", "dev"), cfg.get("max_chars"))
        if key not in cache:
            cache[key] = episodes(key[0], max_chars=key[1])
        eps = cache[key]
        diag = [json.loads(l) for l in (run / "diag.jsonl").open()] if (run / "diag.jsonl").exists() else []
        for f in sorted(run.glob("pools_*.pt"), key=lambda p: (int(p.stem.split("_")[1]), p.stem)):
            label = f.stem[len("pools_"):]
            step = int(label.split("_")[0])
            if (run / f"eval_{label}.json").exists():
                continue
            d = torch.load(f)
            rows, summary = evaluate_pools(d["pool"], d["char_ids"], eps, ext)
            gaps = [r["gap_support_heldout"] for r in diag if r["step"] == step] if label == str(step) else []
            if gaps:
                summary["critic_gap"] = {"mean": float(np.mean(gaps)), "ci": float(1.96 * np.std(gaps) / np.sqrt(len(gaps)))}
            save_rows(rows, summary, run / f"eval_{label}.json")
            print(f"{run.name:28s} {label:>12s} {fmt(summary)}", flush=True)


if __name__ == "__main__":
    main()
