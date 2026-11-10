
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset

from mnist.metrics.classifier import MNISTClassifier
from mnist.metrics.metrics import extract_features, fid, precision_recall


class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x = F.leaky_relu_(self.fc1(x), 0.2)
        x = F.leaky_relu_(self.fc2(x), 0.2)
        x = F.leaky_relu_(self.fc3(x), 0.2)
        x = self.fc4(x)
        return x


def init_weights_uniform(module: nn.Module, clip: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, a=-clip, b=clip)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_weights_normal(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def critic_loss(critic: nn.Module, real_batch: torch.Tensor, fake_batch: torch.Tensor) -> torch.Tensor:
    # One forward pass on both batches: valid because the critic has no batch-dependent layer
    scores = critic(torch.cat([real_batch, fake_batch]))
    n_real = real_batch.size(0)
    return scores[n_real:].mean() - scores[:n_real].mean()

def generator_loss(critic: nn.Module, fake_batch: torch.Tensor) -> torch.Tensor:
    return -critic(fake_batch).mean()

def clip_weights(critic: nn.Module, clip: float):
    with torch.no_grad():
        for p in critic.parameters():
            p.clamp_(-clip, clip)



def subset_to_tensor(subset: Subset, device: torch.device) -> torch.Tensor:
    """Returns a MNIST Subset as a (N, 784) tensor on device, normalized in [-1, 1] like utils.TRANSFORM."""
    images = subset.dataset.data[subset.indices]
    return (images.float() / 255 * 2 - 1).view(-1, 784).to(device)


class IndexSampler:
    """Draws indices without replacement and reshuffles at each epoch, like DataLoader(shuffle=True)."""

    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.device = device
        self.perm = torch.randperm(n, device=device)
        self.position = 0

    def sample(self, size: int) -> torch.Tensor:
        indices = self.perm[self.position : self.position + size]
        self.position += size
        missing = size - indices.size(0)
        if missing > 0:
            self.perm = torch.randperm(self.n, device=self.device)
            self.position = missing
            indices = torch.cat([indices, self.perm[:missing]])
        return indices


@torch.no_grad()
def evaluate_model(classifier: MNISTClassifier, feat_real: np.ndarray, generated: torch.Tensor, k: int, device: torch.device) -> tuple[float, float, float]:
    """Returns FID, precision, recall. feat_real is computed once, the real images never change."""
    feat_fake = extract_features(generated, classifier, batch_size=128, device=device)

    fid_score = fid(feat_real, feat_fake)
    precision, recall = precision_recall(feat_real, feat_fake, k, device)

    return fid_score, precision, recall


@torch.no_grad()
def EM_estimator(critic: nn.Module, real: torch.Tensor, generated: torch.Tensor) -> float:
    return -critic_loss(critic, real, generated).item()
