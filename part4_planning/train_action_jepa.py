"""Action-conditioned JEPA training on the bouncing ball.

Encoder is unchanged. Predictor takes (z_t, action) -> z_{t+1}, where
action is the ball's direction at time t (-1 for left, +1 for right).

The point: with action provided, the bimodal-future ambiguity disappears.
The predictor's job becomes well-defined and the embedding-space MSE
is no longer averaging two valid futures. The encoder no longer has to
guess direction from a single frame; the planner provides it.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.data import make_action_triples
from shared.encoder import Encoder, EMBED_DIM, get_device


class ActionPredictor(nn.Module):
    """Predicts z_{t+1} from (z_t, action) where action is a scalar."""

    def __init__(self, embed_dim: int = EMBED_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = action.unsqueeze(-1) if action.dim() == 1 else action
        return self.net(torch.cat([z, a], dim=-1))


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
    steps: int = 6000,
    batch_size: int = 256,
    lr: float = 3e-4,
    sim_w: float = 25.0,
    var_w: float = 25.0,
    cov_w: float = 1.0,
    seed: int = 0,
):
    device = get_device()
    print(f"device: {device}")

    pairs, actions = make_action_triples(num_pairs=4096, seed=seed)
    pairs = pairs.to(device)
    actions = actions.to(device)
    print(f"pairs: {tuple(pairs.shape)}, actions: {tuple(actions.shape)}")

    encoder = Encoder().to(device)
    predictor = ActionPredictor().to(device)
    n_params = sum(p.numel() for p in encoder.parameters()) + sum(
        p.numel() for p in predictor.parameters()
    )
    print(f"trainable params: {n_params:,}")

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr
    )

    rng = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, pairs.shape[0], (batch_size,), generator=rng)
        x_t, x_next = pairs[idx, 0], pairs[idx, 1]
        a_t = actions[idx]

        z_t = encoder(x_t)
        z_next = encoder(x_next)
        z_pred = predictor(z_t, a_t)

        sim = F.mse_loss(z_pred, z_next)
        var = variance_loss(z_t) + variance_loss(z_next)
        cov = covariance_loss(z_t) + covariance_loss(z_next)
        loss = sim_w * sim + var_w * var + cov_w * cov

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 500 == 0:
            std = z_next.std(dim=0).mean().item()
            print(
                f"step {step + 1:>5d}  sim {sim.item():.4f}  var {var.item():.4f}  "
                f"cov {cov.item():.4f}  z_std {std:.3f}"
            )
    print(f"trained in {time.time() - t0:.1f}s")

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
        },
        "checkpoints/action_jepa.pt",
    )
    print("saved checkpoints/action_jepa.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    args = parser.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    train(steps=args.steps)
