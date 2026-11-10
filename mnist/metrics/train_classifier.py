
import argparse
import json
import time
from pathlib import Path

import torch
from mnist.metrics.classifier import MNISTClassifier
from torch import nn
from torch.utils.data import DataLoader

from mnist.utils import get_mnist_splits


def init_weights(module: nn.Module) -> None:
    """Truncated normal (std 0.1, cut at 2 std), zero biases.

    Args:
        module: The module to initialize.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.1, a=-0.2, b=0.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate the model on a given data loader (loss, accuracy).

    Args:
        model: The model to evaluate.
        loader: DataLoader for the evaluation dataset.
        device: Device to perform computations on.
    """

    model.eval()
    total_loss, correct, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += nn.CrossEntropyLoss(reduction='sum')(logits, y).item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.numel()
    return total_loss / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MNIST feature-extractor classifier.")
    parser.add_argument("--data_path", type=str, default="data/mnist", help="Path to the MNIST dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay (L2 regularization) for the optimizer.")
    parser.add_argument("--decay_rate", type=float, default=0.1, help="Learning rate decay factor per decay_steps steps.")
    parser.add_argument("--decay_steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--run_name", type=str, default="classifier_mnist")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")


    train_dataset, val_dataset = get_mnist_splits(Path(args.data_path), download=True, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")

    out_dir = Path("runs").joinpath(Path(args.run_name))
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    else:
        raise FileExistsError(f"{out_dir} already exists")
    
    model = MNISTClassifier().to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate**(1 / args.decay_steps))

    best_path = out_dir.joinpath("best.pth")
    last_path = out_dir.joinpath("last.pth")
    with open(out_dir.joinpath("config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
   
    best_acc, history = 0.0, []

    for epoch in range(1, args.epochs+1):
        model.train()
        start = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.CrossEntropyLoss(reduction="mean")(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss, train_acc = evaluate_model(model, train_loader, device)
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, 
               "val_loss": val_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0], 
               "seconds": time.time() - start}
        history.append(row)
        print(json.dumps(row))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "best_val_acc": best_acc},
                    last_path)

    summary = {"best_val_accuracy": best_acc,
               "best_epoch": max(history, key=lambda r: r["val_acc"])["epoch"]}
    with open(out_dir.joinpath("summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best validation accuracy: {best_acc:.4f} -> {best_path}")

if __name__ == "__main__":
    main()