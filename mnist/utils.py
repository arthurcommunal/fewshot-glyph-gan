import os
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

def get_mnist_splits(datapath: Path, download: bool, seed: int = 42, n_val: int = 10000) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Get MNIST train and validation splits.

    Args:
        datapath: Path to the MNIST dataset.
        download: Whether to download the dataset if not present.
        seed: Random seed for reproducibility.
        n_val: Number of validation samples.
    """

    if not datapath.exists():
        datapath.mkdir(parents=True)

    full_train = datasets.MNIST(root=datapath, train=True, transform=TRANSFORM, download=download)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [n_train, n_val], generator=generator)
    return train, val


def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder, device):
    ckpt_path = os.path.join(folder,'G.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
