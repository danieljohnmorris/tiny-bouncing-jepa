"""Pixel-space baseline.

Encoder + decoder trained end-to-end to predict the next frame in pixel
space using MSE loss. On the bouncing-ball dataset, wall-bounce frames are
ambiguous (the next frame is equally likely to bounce left or right), so
MSE forces the model to predict the average — a blur.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.data import make_pairs
from shared.encoder import Encoder, Decoder, get_device


class PixelPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train(steps: int = 2000, batch_size: int = 64, lr: float = 3e-4, seed: int = 0):
    device = get_device()
    print(f"device: {device}")

    pairs = make_pairs(num_pairs=4096, seed=seed).to(device)  # (N, 2, 1, 64, 64)
    print(f"pairs: {tuple(pairs.shape)}")

    model = PixelPredictor().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    rng = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, pairs.shape[0], (batch_size,), generator=rng)
        batch = pairs[idx]
        x_t, x_next = batch[:, 0], batch[:, 1]
        pred = model(x_t)
        loss = F.mse_loss(pred, x_next)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 200 == 0:
            print(f"step {step + 1:>5d}  loss {loss.item():.5f}")
    print(f"trained in {time.time() - t0:.1f}s")

    torch.save({"model": model.state_dict(), "losses": losses}, "checkpoints/pixel.pt")
    print("saved checkpoints/pixel.pt")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    import os
    os.makedirs("checkpoints", exist_ok=True)
    train(steps=args.steps)
