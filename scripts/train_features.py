"""Train the metric feature extractor on the background split, then sanity-check it on dev episodes.

    python scripts/train_features.py --out runs/features/clf.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glyphgan.data import episodes, load_split  # noqa: E402
from glyphgan.evaluate import alphabet_protos  # noqa: E402
from glyphgan.features import DIM, FeatureExtractor, FeatureNet, augment  # noqa: E402


def holdout_mask(char: torch.Tensor, per_char: int, seed: int) -> torch.Tensor:
    """True for `per_char` random drawings of every character (validation)."""
    rng, val = np.random.default_rng(seed), torch.zeros(len(char), dtype=torch.bool)
    for c in char.unique():
        idx = (char == c).nonzero().squeeze(1).numpy()
        val[rng.choice(idx, per_char, replace=False)] = True
    return val


@torch.no_grad()
def accuracy(net: FeatureNet, x: torch.Tensor, y: torch.Tensor) -> float:
    net.eval()
    return (torch.cat([net.logits(x[i:i + 4096]).argmax(1) for i in range(0, len(x), 4096)]) == y).float().mean().item()


def proto_accuracy(ext: FeatureExtractor, part: str = "dev") -> float:
    """Nearest-prototype accuracy: prototypes = mean support embedding of each character of the alphabet,
    queries = heldout drawings; mean over characters."""
    eps = episodes(part)
    protos = alphabet_protos(eps, [ext(ep.support) for ep in eps])
    accs = []
    for ep in eps:
        P, chars = protos[ep.alphabet]
        pred = (ext(ep.heldout) @ F.normalize(P, dim=1).T).argmax(1)
        accs.append((pred == chars.index(ep.char)).float().mean().item())
    return float(np.mean(accs))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/features/clf.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    bg = load_split("data", "background")
    val = holdout_mask(bg.char, 2, a.seed)
    xtr, ytr = bg.x[~val].to(dev), bg.char[~val].to(dev)
    xva, yva = bg.x[val].to(dev), bg.char[val].to(dev)
    n_classes = int(bg.char.max()) + 1
    net = FeatureNet(n_classes, DIM).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=5e-4)
    steps = a.epochs * ((len(xtr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    for ep in range(a.epochs):
        net.train()
        perm, tot = torch.randperm(len(xtr), device=dev), 0.0
        for i in range(0, len(xtr), a.bs):
            b = perm[i:i + a.bs]
            loss = F.cross_entropy(net.logits(augment(xtr[b])), ytr[b], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"epoch {ep + 1} loss {tot / len(xtr):.3f} train_acc {accuracy(net, xtr, ytr):.4f} "
                  f"val_acc {accuracy(net, xva, yva):.4f}", flush=True)

    val_acc = accuracy(net, xva, yva)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.cpu().state_dict(), "n_classes": n_classes, "dim": DIM, "val_acc": val_acc,
                "config": vars(a)}, a.out)
    ext = FeatureExtractor.load(a.out, dev)
    print(f"saved {a.out} | val_acc {val_acc:.4f} | dev proto_acc {proto_accuracy(ext, 'dev'):.4f}")


if __name__ == "__main__":
    main()
