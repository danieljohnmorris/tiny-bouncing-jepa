"""MPC-style planner in JEPA's latent space.

Given a current frame x_0 and a goal frame x_g, find a sequence of
actions [a_0, ..., a_{H-1}] such that rolling out the action-conditioned
predictor from encoder(x_0) lands close to encoder(x_g).

Action space is {-1, +1}, horizon H is small, so we just enumerate all
2^H sequences and pick the one with lowest L2 distance to the goal
embedding. No gradient-based optimization needed.

We then visualise the best plan as a frame strip: ground-truth episode
on top, model rollout decoded back to pixels via the JEPA visualisation
decoder on the bottom.

The decoder used here is the per-frame autoencoder decoder (not the
predictor-output decoder from Part 2), because we want to render
encoder outputs along the planned trajectory, not predictor averages.
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.data import simulate_episode, FRAME_SIZE
from shared.encoder import Encoder, Decoder, get_device
from part4_planning.train_action_jepa import ActionPredictor


def load_models(device):
    j = torch.load("checkpoints/action_jepa.pt", map_location=device, weights_only=True)
    encoder = Encoder().to(device)
    encoder.load_state_dict(j["encoder"])
    predictor = ActionPredictor().to(device)
    predictor.load_state_dict(j["predictor"])
    encoder.train(False)
    predictor.train(False)
    for p in encoder.parameters(): p.requires_grad_(False)
    for p in predictor.parameters(): p.requires_grad_(False)
    return encoder, predictor


def train_autoenc_decoder(encoder, device, steps: int = 4000, seed: int = 0):
    """Train a decoder to reconstruct frame_t from encoder(frame_t).

    Used purely for visualisation: lets us render embeddings along the
    planned rollout back to pixels.
    """
    from shared.data import make_pairs
    pairs = make_pairs(num_pairs=8192, seed=seed).to(device)
    frames = pairs.view(-1, 1, 64, 64)

    decoder = Decoder().to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    print(f"training autoencoder decoder ({steps} steps)")
    for step in range(steps):
        idx = torch.randint(0, frames.shape[0], (128,), generator=rng)
        x = frames[idx]
        with torch.no_grad():
            z = encoder(x)
        recon = decoder(z)
        loss = F.mse_loss(recon, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  step {step + 1:>5d}  loss {loss.item():.5f}")
    decoder.train(False)
    for p in decoder.parameters(): p.requires_grad_(False)
    return decoder


def enumerate_actions(horizon: int) -> torch.Tensor:
    """All 2^horizon action sequences with values in {-1, +1}. Shape (2^H, H)."""
    n = 2 ** horizon
    bits = torch.tensor([[(i >> b) & 1 for b in range(horizon)] for i in range(n)], dtype=torch.float32)
    return 2 * bits - 1


def plan(encoder, predictor, x_0: torch.Tensor, x_g: torch.Tensor, horizon: int) -> torch.Tensor:
    """Return best action sequence as a (horizon,) tensor."""
    device = x_0.device
    with torch.no_grad():
        z_0 = encoder(x_0[None, None]) if x_0.dim() == 2 else encoder(x_0)
        z_g = encoder(x_g[None, None]) if x_g.dim() == 2 else encoder(x_g)

    seqs = enumerate_actions(horizon).to(device)  # (N, H)
    n = seqs.shape[0]

    # Roll out each candidate from z_0 and score by terminal distance to z_g.
    z = z_0.expand(n, -1).clone()
    for t in range(horizon):
        a = seqs[:, t]
        z = predictor(z, a)
    dist = ((z - z_g.expand(n, -1)) ** 2).sum(dim=-1)

    best = int(dist.argmin().item())
    return seqs[best]


def upscale(img: np.ndarray, factor: int = 4) -> np.ndarray:
    return img.repeat(factor, axis=0).repeat(factor, axis=1)


def render_strip(frames: list[np.ndarray], gap: int = 8) -> np.ndarray:
    h = frames[0].shape[0]
    sep = np.full((h, gap), 0.4, dtype=np.float32)
    cols = []
    for i, f in enumerate(frames):
        cols.append(f)
        if i < len(frames) - 1:
            cols.append(sep)
    return np.concatenate(cols, axis=1)


def main(args):
    device = get_device()
    print(f"device: {device}")

    encoder, predictor = load_models(device)
    decoder = train_autoenc_decoder(encoder, device, steps=args.decoder_steps)

    rng = np.random.default_rng(args.seed)
    horizon = args.horizon
    episode = simulate_episode(steps=horizon + 1, rng=rng)
    x_0_np = episode[0]
    x_g_np = episode[-1]

    x_0 = torch.from_numpy(x_0_np).to(device)
    x_g = torch.from_numpy(x_g_np).to(device)

    actions = plan(encoder, predictor, x_0, x_g, horizon=horizon)
    print(f"planned actions: {actions.cpu().numpy().tolist()}")

    # Roll out using planned actions, decode each step.
    z = encoder(x_0[None, None])
    decoded_rollout = [decoder(z).cpu().numpy()[0, 0]]
    for t in range(horizon):
        z = predictor(z, actions[t:t + 1])
        decoded_rollout.append(decoder(z).cpu().numpy()[0, 0])

    # Render: top row = ground truth episode, bottom row = decoded planned rollout.
    gt_row = render_strip([upscale(ep) for ep in episode])
    plan_row = render_strip([upscale(f) for f in decoded_rollout])
    bar = np.full((8, gt_row.shape[1]), 0.6, dtype=np.float32)
    full = np.concatenate([gt_row, bar, plan_row], axis=0)

    os.makedirs("part4_planning/outputs", exist_ok=True)
    out_path = "part4_planning/outputs/plan_strip.png"
    imageio.imwrite(out_path, (np.clip(full, 0, 1) * 255).astype(np.uint8))
    print(f"wrote {out_path}")

    # Also report final L2 to goal in pixel space.
    final_l2 = float(np.sqrt(((decoded_rollout[-1] - x_g_np) ** 2).mean()))
    print(f"final pixel-space RMS distance to goal: {final_l2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--decoder-steps", type=int, default=4000)
    args = parser.parse_args()
    main(args)
