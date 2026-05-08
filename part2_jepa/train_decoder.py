"""Train a small decoder on the JEPA predictor's output embeddings.

The JEPA model itself never produces pixels — it predicts embeddings.
To visualize what its predictions look like, we train a decoder to
invert the predictor's output back to the next frame.

This matches the visualization use case: visualize.py feeds
predictor(encoder(x_t)) to the decoder, so the decoder must be trained
on that exact distribution. Training on encoder(x) alone produces
embeddings the decoder never sees at inference time.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.data import make_pairs, make_pairs_noisy_input_clean_target
from shared.encoder import Encoder, Decoder, get_device
from part2_jepa.train_jepa import Predictor


def train(steps: int = 1500, batch_size: int = 128, lr: float = 3e-4, seed: int = 0, noisy: bool = False):
    device = get_device()
    print(f"device: {device}, noisy: {noisy}")

    jepa_path = "checkpoints/jepa_noisy.pt" if noisy else "checkpoints/jepa.pt"
    ckpt = torch.load(jepa_path, map_location=device, weights_only=True)
    encoder = Encoder().to(device)
    encoder.load_state_dict(ckpt["encoder"])
    predictor = Predictor().to(device)
    predictor.load_state_dict(ckpt["predictor"])
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in predictor.parameters():
        p.requires_grad_(False)

    decoder = Decoder().to(device)
    print(f"decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    num_pairs = 32768
    if noisy:
        # Encoder/predictor were trained on noisy inputs so x_t stays noisy,
        # but x_{t+1} is clean here so the decoder learns to reconstruct the
        # predictable signal (ball position) without hallucinating fresh noise.
        pairs = make_pairs_noisy_input_clean_target(num_pairs=num_pairs, seed=seed).to(device)
    else:
        pairs = make_pairs(num_pairs=num_pairs, seed=seed).to(device)
    x_t = pairs[:, 0]      # (N, 1, H, W) - noisy if --noisy
    x_next = pairs[:, 1]   # (N, 1, H, W) - always clean if --noisy
    print(f"pairs: {tuple(pairs.shape)}")

    opt = torch.optim.AdamW(decoder.parameters(), lr=lr)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, x_t.shape[0], (batch_size,), generator=rng)
        with torch.no_grad():
            z_t = encoder(x_t[idx])
            z_pred = predictor(z_t)
        recon = decoder(z_pred)
        loss = F.mse_loss(recon, x_next[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            print(f"step {step + 1:>5d}  loss {loss.item():.5f}")
    print(f"trained in {time.time() - t0:.1f}s")

    out_path = "checkpoints/jepa_decoder_noisy.pt" if noisy else "checkpoints/jepa_decoder.pt"
    torch.save({"decoder": decoder.state_dict()}, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--noisy", action="store_true")
    args = parser.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    train(steps=args.steps, noisy=args.noisy)
