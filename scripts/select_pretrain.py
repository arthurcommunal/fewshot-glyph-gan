"""Score pretraining checkpoints: FID + precision/recall/density/coverage (k=5) of 10k G_ema samples, in the Omniglot
feature space, against (a) background images (what G was trained on) and (b) dev-evaluation images (unseen alphabets:
how general the learned prior of strokes is).

    python -m scripts.select_pretrain runs/pretrain/gp_a runs/pretrain/gp_b ...

Writes <run>/select.json and prints one line per checkpoint.
"""
import json
import sys
from pathlib import Path

import torch

from glyphgan.data import eval_alphabets, load_split
from glyphgan.features import FeatureExtractor
from glyphgan.metrics import prdc
from glyphgan.models import Z_DIM, new_generator
from mnist.metrics.metrics import fid

N = 10_000


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    ext = FeatureExtractor.load("runs/features/clf.pt", dev)
    g = torch.Generator().manual_seed(0)
    bg = load_split("data", "background").x
    ev = load_split("data", "evaluation")
    keep = torch.isin(ev.alphabet, torch.tensor(eval_alphabets("dev")))
    refs = {"bg": ext(bg[torch.randperm(len(bg), generator=g)[:N]].to(dev)).cpu(),
            "dev": ext(ev.x[keep].to(dev)).cpu()}
    z = torch.randn(N, Z_DIM, generator=g).to(dev)
    for run in map(Path, sys.argv[1:]):
        res = []
        for f in sorted(run.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
            ck = torch.load(f, map_location="cpu")
            row = {"step": ck["step"]}
            for w in ("G_ema", "G"):
                G = new_generator(); G.load_state_dict(ck[w]); G.to(dev).eval()
                feat = ext(G(z)).cpu()
                for name, ref in refs.items():
                    row[f"{w}_{name}_fid"] = fid(ref.numpy(), feat.numpy())
                    m = prdc(ref[:5000], feat[:5000], k=5)
                    row.update({f"{w}_{name}_{k}": v for k, v in m.items()})
            res.append(row)
            print(run.name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()
                                        if k.startswith("G_ema") or k == "step"}), flush=True)
        (run / "select.json").write_text(json.dumps(res, indent=2))
        best = min(res, key=lambda r: r["G_ema_bg_fid"])
        print(f"BEST {run.name} step={best['step']} bg_fid={best['G_ema_bg_fid']:.3f} dev_fid={best['G_ema_dev_fid']:.3f}", flush=True)


if __name__ == "__main__":
    main()
