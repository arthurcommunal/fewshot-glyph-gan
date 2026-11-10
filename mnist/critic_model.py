"""Critic used at inference time to refine the latents (the generator itself is unchanged).

Same architecture and parameter names as the critic team's CNNCritic (critic/lab.py, eval/evlib.py):
DCGAN-like, no batch norm, trained post hoc with a gradient penalty on the frozen generator."""
import torch
from torch import nn


class CNNCritic(nn.Module):
    def __init__(self, w: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, w, 4, 2, 1), nn.LeakyReLU(0.2),          # 28 -> 14
            nn.Conv2d(w, 2 * w, 4, 2, 1), nn.LeakyReLU(0.2),      # 14 -> 7
            nn.Conv2d(2 * w, 4 * w, 3, 2, 1), nn.LeakyReLU(0.2),  # 7 -> 4
            nn.Flatten(), nn.Linear(4 * w * 16, 1))

    def forward(self, x):
        return self.net(x.view(-1, 1, 28, 28))


def load_critic(path, device) -> CNNCritic:
    """Accepts a plain state_dict or {'kind': 'cnn', 'state_dict': ...}; the width is read from the weights."""
    sd = torch.load(path, map_location="cpu")
    if "state_dict" in sd:
        if sd.get("kind", "cnn") != "cnn":
            raise ValueError(f"critic kind {sd.get('kind')!r} not supported (cnn only)")
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    critic = CNNCritic(w=sd["net.0.weight"].shape[0])
    critic.load_state_dict(sd)
    critic.to(device).eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    return critic
