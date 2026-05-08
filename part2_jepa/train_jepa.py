"""JEPA-style next-frame predictor with VICReg loss.

A single encoder maps frame_t and frame_t+1 into a shared embedding space.
A small predictor MLP maps embedding(t) to embedding(t+1). Training uses
the VICReg objective (Bardes, Ponce, LeCun, 2022):

  - invariance: predictor output should match the next-frame embedding (MSE)
  - variance:   each embedding dimension must have stddev >= 1 across the batch
  - covariance: off-diagonal entries of the embedding covariance matrix are
                pushed toward zero, decorrelating the dimensions

The variance term prevents the trivial "output a constant for everything"
collapse that pure invariance loss would learn. The covariance term stops
all the information piling into a few dimensions.

The point: in embedding space, we don't have to commit to a single
pixel-level future. The encoder is free to discard unpredictable details
(which way the next bounce goes) and keep what's predictable. No averaging,
no blur.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.data import make_pairs, make_pairs_noisy
from shared.encoder import Encoder, EMBED_DIM, get_device


class Predictor(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def variance_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(target_std - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / d


def train(
    steps: int = 3000,
    batch_size: int = 256,
    lr: float = 3e-4,
    sim_w: float = 25.0,
    var_w: float = 25.0,
    cov_w: float = 1.0,
    seed: int = 0,
    noisy: bool = False,
):
    device = get_device()
    print(f"device: {device}, noisy: {noisy}")

    if noisy:
        pairs = make_pairs_noisy(num_pairs=4096, seed=seed).to(device)
    else:
        pairs = make_pairs(num_pairs=4096, seed=seed).to(device)
    print(f"pairs: {tuple(pairs.shape)}")

    encoder = Encoder().to(device)
    predictor = Predictor().to(device)

    n_params = sum(p.numel() for p in encoder.parameters()) + sum(
        p.numel() for p in predictor.parameters()
    )
    print(f"trainable params: {n_params:,}")

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr
    )

    rng = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, pairs.shape[0], (batch_size,), generator=rng)
        batch = pairs[idx]
        x_t, x_next = batch[:, 0], batch[:, 1]

        z_t = encoder(x_t)
        z_next = encoder(x_next)
        z_pred = predictor(z_t)

        sim = F.mse_loss(z_pred, z_next)
        var = variance_loss(z_t) + variance_loss(z_next)
        cov = covariance_loss(z_t) + covariance_loss(z_next)
        loss = sim_w * sim + var_w * var + cov_w * cov

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append({"sim": sim.item(), "var": var.item(), "cov": cov.item()})
        if (step + 1) % 200 == 0:
            std = z_next.std(dim=0).mean().item()
            print(
                f"step {step + 1:>5d}  sim {sim.item():.4f}  var {var.item():.4f}  "
                f"cov {cov.item():.4f}  z_std {std:.3f}"
            )
    print(f"trained in {time.time() - t0:.1f}s")

    out_path = "checkpoints/jepa_noisy.pt" if noisy else "checkpoints/jepa.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "losses": losses,
        },
        out_path,
    )
    print(f"saved {out_path}")
    return encoder, predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--noisy", action="store_true")
    args = parser.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    train(steps=args.steps, noisy=args.noisy)
