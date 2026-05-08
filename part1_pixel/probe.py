"""Probe pixel-MSE vs JEPA encoders for ball-position prediction.

Both encoders map (B, 1, 64, 64) -> (B, 128). They were trained for
different objectives:

  - pixel encoder: end-to-end MSE on next-frame pixel reconstruction
  - JEPA encoder: VICReg + similarity to predictor output (no pixels)

A linear probe trained on top of each frozen encoder asks: how well
does this representation linearly encode ball x-position? This is the
honest version of the LeCun argument - the visual smear is a property
of MSE on bimodal futures (both encoders smear when decoded), but the
JEPA encoder's representation can still be more useful for downstream
tasks. This probe measures that directly.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.data import simulate_episode, FRAME_SIZE, BALL_RADIUS
from shared.encoder import Encoder, get_device


def make_position_dataset(num_frames: int, seed: int):
    """Generate frames paired with their ball x-position (normalised to [-1, 1])."""
    rng = np.random.default_rng(seed)
    frames = []
    positions = []
    while len(frames) < num_frames:
        episode = simulate_episode(steps=24, rng=rng)
        for t in range(len(episode)):
            frames.append(episode[t])
            col_mass = episode[t].sum(axis=0)
            x = float((col_mass * np.arange(FRAME_SIZE)).sum() / col_mass.sum())
            x_norm = (x - FRAME_SIZE / 2) / (FRAME_SIZE / 2 - BALL_RADIUS - 1)
            positions.append(x_norm)
            if len(frames) >= num_frames:
                break
    x = torch.from_numpy(np.stack(frames)[:, None, :, :].astype(np.float32))
    y = torch.from_numpy(np.array(positions, dtype=np.float32))[:, None]
    return x, y


def load_pixel_encoder(device):
    enc = Encoder().to(device)
    ckpt = torch.load("checkpoints/pixel.pt", map_location=device, weights_only=True)
    state = {k.removeprefix("encoder."): v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
    enc.load_state_dict(state)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.train(False)
    return enc


def load_jepa_encoder(device):
    enc = Encoder().to(device)
    ckpt = torch.load("checkpoints/jepa.pt", map_location=device, weights_only=True)
    enc.load_state_dict(ckpt["encoder"])
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.train(False)
    return enc


def train_probe(encoder, x_train, y_train, x_test, y_test, device, steps: int = 2000, lr: float = 1e-3):
    """Train a linear probe on top of a frozen encoder. Returns final test MSE."""
    with torch.no_grad():
        z_train = encoder(x_train.to(device))
        z_test = encoder(x_test.to(device))
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    probe = nn.Linear(z_train.shape[1], 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    for step in range(steps):
        idx = torch.randint(0, z_train.shape[0], (256,), device=device)
        pred = probe(z_train[idx])
        loss = F.mse_loss(pred, y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.train(False)
    with torch.no_grad():
        test_pred = probe(z_test)
        test_mse = F.mse_loss(test_pred, y_test).item()
    return test_mse


def main(num_train: int = 4096, num_test: int = 1024, steps: int = 2000, seed: int = 0):
    device = get_device()
    print(f"device: {device}\n")

    print(f"generating dataset: {num_train} train + {num_test} test frames")
    x_train, y_train = make_position_dataset(num_train, seed=seed)
    x_test, y_test = make_position_dataset(num_test, seed=seed + 1)
    print(f"train frames: {tuple(x_train.shape)}, labels: {tuple(y_train.shape)}\n")

    print("loading encoders")
    pixel_enc = load_pixel_encoder(device)
    jepa_enc = load_jepa_encoder(device)

    print(f"\ntraining linear probe on pixel-MSE encoder ({steps} steps)")
    t0 = time.time()
    pixel_mse = train_probe(pixel_enc, x_train, y_train, x_test, y_test, device, steps=steps)
    print(f"  pixel test MSE: {pixel_mse:.5f}  (in {time.time() - t0:.1f}s)")

    print(f"\ntraining linear probe on JEPA encoder ({steps} steps)")
    t0 = time.time()
    jepa_mse = train_probe(jepa_enc, x_train, y_train, x_test, y_test, device, steps=steps)
    print(f"  jepa  test MSE: {jepa_mse:.5f}  (in {time.time() - t0:.1f}s)")

    print(f"\nratio: pixel/jepa = {pixel_mse / jepa_mse:.2f}x")
    if jepa_mse < pixel_mse:
        print("-> JEPA representation is more linearly informative for ball position")
    else:
        print("-> pixel-MSE representation is more linearly informative for ball position")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=4096)
    parser.add_argument("--num-test", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    os.makedirs("part1_pixel/outputs", exist_ok=True)
    main(num_train=args.num_train, num_test=args.num_test, steps=args.steps)
