"""Train a small decoder on the JEPA encoder's frozen embeddings.

The JEPA model itself never produces pixels — it predicts embeddings. To
*visualize* what its predictions look like, we need to invert the encoder.

We freeze the trained encoder, sample bouncing-ball frames, and train a
decoder to reconstruct each frame from its embedding. This decoder is
purely for visualization; the JEPA model never sees it during training,
and it has nothing to do with the actual learning task.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.data import make_pairs
from shared.encoder import Encoder, Decoder, get_device


def train(steps: int = 1500, batch_size: int = 128, lr: float = 3e-4, seed: int = 0):
    device = get_device()
    print(f"device: {device}")

    ckpt = torch.load("checkpoints/jepa.pt", map_location=device, weights_only=True)
    encoder = Encoder().to(device)
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad_(False)

    decoder = Decoder().to(device)
    print(f"decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    pairs = make_pairs(num_pairs=4096, seed=seed).to(device)
    frames = pairs.view(-1, 1, 64, 64)
    print(f"frames: {tuple(frames.shape)}")

    opt = torch.optim.AdamW(decoder.parameters(), lr=lr)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, frames.shape[0], (batch_size,), generator=rng)
        x = frames[idx]
        with torch.no_grad():
            z = encoder(x)
        recon = decoder(z)
        loss = F.mse_loss(recon, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            print(f"step {step + 1:>5d}  loss {loss.item():.5f}")
    print(f"trained in {time.time() - t0:.1f}s")

    torch.save({"decoder": decoder.state_dict()}, "checkpoints/jepa_decoder.pt")
    print("saved checkpoints/jepa_decoder.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    train(steps=args.steps)
