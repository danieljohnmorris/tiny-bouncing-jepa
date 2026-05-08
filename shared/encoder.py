"""Tiny ConvNet encoder shared across all training scripts.

Maps (B, 1, 64, 64) -> (B, embed_dim). About 200K parameters.
"""

import torch
import torch.nn as nn

EMBED_DIM = 128


class Encoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16x16
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8x8
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Inverse of the encoder, used for visualization only."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.lift = nn.Linear(embed_dim, 8 * 8 * 64)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.lift(z).view(-1, 64, 8, 8)
        return self.net(x)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
