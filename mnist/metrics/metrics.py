"""Precision, recall and FID metrics"""

import numpy as np
import torch
from scipy import linalg
from sklearn.neighbors import NearestNeighbors

from mnist.metrics.classifier import MNISTClassifier


@torch.no_grad()
def extract_features(images: torch.Tensor | np.ndarray, classifier: MNISTClassifier, batch_size: int, device: torch.device) -> np.ndarray:
    """Extract features of images with the trained MNIST classifier.

    Args:
        images: Images normalized in [-1, 1], of shape (N, 784), (N, 28, 28) or (N, 1, 28, 28).
        classifier: Trained MNISTClassifier.
        batch_size: Number of images per forward pass.
        device: Device to use for the classifier.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    # Reshape to (N, 1, 28, 28), the input format of the classifier
    if images.numel() % (28 * 28) != 0:
        raise ValueError(f"Cannot reshape images of shape {tuple(images.shape)} to (N, 1, 28, 28).")
    images = images.float().reshape(-1, 1, 28, 28)

    # The classifier was trained on images normalized in [-1, 1] (same range as the generator's tanh)
    if images.min() < -1 - 1e-3 or images.max() > 1 + 1e-3:
        raise ValueError(f"Images must be in [-1, 1], got [{images.min():.3f}, {images.max():.3f}].")

    classifier = classifier.to(device).eval()
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        features.append(classifier.extract_features(batch).cpu())

    return torch.cat(features).numpy() # (N, 84)


def precision_recall(feat_real: np.ndarray, feat_fake: np.ndarray, k: int, device: torch.device | None = None) -> tuple[float, float]:
    """Compute precision and recall metrics with Kynkäänniemi estimators.

    Args:
        feat_real: Features of real images.
        feat_fake: Features of fake images.
        k: Number of nearest neighbors to consider.
        device: Device to use for tensor computations.
    """

    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if device is None:
        device = torch.device("cpu")
    
    # Fit nearest neighbors on real features and fake features
    nbrs_real = NearestNeighbors(n_neighbors=k+1).fit(feat_real)
    nbrs_fake = NearestNeighbors(n_neighbors=k+1).fit(feat_fake)

    # Compute distances, precision and recall per batch
    distances_real, _ = nbrs_real.kneighbors(feat_real, n_neighbors=k+1) # (n_reals, k+1)
    distances_real = torch.from_numpy(distances_real).float().to(device)
    distances_fake, _ = nbrs_fake.kneighbors(feat_fake, n_neighbors=k+1) # (n_fakes, k+1)
    distances_fake = torch.from_numpy(distances_fake).float().to(device)    
    
    computed = 0 
    B = 1000
    n_fakes = len(feat_fake)
    n_reals = len(feat_real)
    
    fake_tensor = torch.from_numpy(feat_fake).float().to(device)
    real_tensor = torch.from_numpy(feat_real).float().to(device)
    
    precision_list = torch.zeros(n_fakes, dtype=torch.bool, device=device)
    recall_list = torch.zeros(n_reals, dtype=torch.bool, device=device)
    
    while computed < n_fakes:
        batch_size = min(B, n_fakes - computed)
        cross_distances = torch.cdist(fake_tensor[computed : computed + batch_size], real_tensor, p=2) # (batch_size, n_reals)
        precision_batch = torch.any(cross_distances <= distances_real[:, -1].reshape(1, -1), dim=1) # (batch_size,)
        recall_batch = torch.any(cross_distances <= distances_fake[computed : computed+batch_size, -1].reshape(-1, 1), dim=0) # (n_reals,)
        precision_list[computed : computed + batch_size] = precision_batch
        recall_list |= recall_batch
        computed += batch_size

    # Final precision and recall
    precision = torch.mean(precision_list.float()).item()
    recall = torch.mean(recall_list.float()).item()

    return precision, recall


def fid(feat_real: np.ndarray, feat_fake: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the FID: Wasserstein-2 distance between the Gaussians fitted on real and fake features.

    FID = ||mu_r - mu_f||^2 + Tr(S_r + S_f - 2 (S_r S_f)^(1/2))

    Args:
        feat_real: Features of real images, of shape (n_reals, d).
        feat_fake: Features of fake images, of shape (n_fakes, d).
        eps: Offset added to the covariance diagonals if their product is singular.
    """

    if feat_real.ndim != 2 or feat_fake.ndim != 2 or feat_real.shape[1] != feat_fake.shape[1]:
        raise ValueError(f"Features must be of shape (n, d) with the same d, got {feat_real.shape} and {feat_fake.shape}.")
    if len(feat_real) < 2 or len(feat_fake) < 2:
        raise ValueError("At least 2 samples per distribution are needed to estimate a covariance.")

    # Fit the Gaussians (float64 for the numerical stability of sqrtm)
    feat_real = feat_real.astype(np.float64)
    feat_fake = feat_fake.astype(np.float64)
    mu_real, mu_fake = feat_real.mean(axis=0), feat_fake.mean(axis=0) # (d,)
    sigma_real = np.cov(feat_real, rowvar=False) # (d, d)
    sigma_fake = np.cov(feat_fake, rowvar=False) # (d, d)

    # Square root of the covariance product
    covmean = linalg.sqrtm(sigma_real @ sigma_fake)
    if not np.isfinite(covmean).all():
        # Singular product (e.g. dead ReLU features): offset the diagonals
        offset = eps * np.eye(sigma_real.shape[0])
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_fake + offset))

    # sqrtm may return a complex matrix with negligible imaginary parts due to numerical errors
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(np.diagonal(covmean)), 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(np.imag(covmean))):.3g} too large in sqrtm.")
        covmean = np.real(covmean)

    diff = mu_real - mu_fake
    return float(diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean))