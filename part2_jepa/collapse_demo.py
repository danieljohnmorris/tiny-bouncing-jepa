"""Representation collapse, demonstrated and fixed.

Train two copies of a tiny encoder on pairs of augmented frames. The naive
loss (just minimize embedding distance between the two views) collapses to
the trivial all-constant solution: the encoder maps every input to the same
vector, achieving zero loss.

The Barlow Twins fix (Zbontar et al., 2021): compute the cross-correlation
matrix between the two batches of embeddings (one row per dim of view A,
one column per dim of view B). Push the matrix toward the identity:
on-diagonal entries to 1, off-diagonal entries to 0. The off-diagonal
penalty forces decorrelation, which alone is enough to prevent collapse.

We log a 2D PCA of the embeddings and the cross-correlation matrix at
intervals during training, then render an animated comparison.
"""

import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.data import simulate_episode
from shared.encoder import Encoder, EMBED_DIM, get_device


def gather_frames(num_episodes: int = 200, steps: int = 24, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    eps = [simulate_episode(steps=steps, rng=rng) for _ in range(num_episodes)]
    arr = np.concatenate(eps)[:, None, :, :]
    return torch.from_numpy(arr)


def augment(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Two-view augmentation: small spatial shift + intensity jitter."""
    b = x.shape[0]
    out = torch.zeros_like(x)
    shifts = torch.randint(-3, 4, (b, 2), generator=rng)
    intensity = 0.7 + 0.3 * torch.rand(b, generator=rng)
    for i in range(b):
        sx, sy = int(shifts[i, 0]), int(shifts[i, 1])
        shifted = torch.roll(x[i], shifts=(sx, sy), dims=(-2, -1))
        out[i] = shifted * intensity[i]
    return out


def barlow_loss(z1: torch.Tensor, z2: torch.Tensor, off_w: float = 0.005) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (total_loss, cross_corr_matrix)."""
    n, d = z1.shape
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-5)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-5)
    cc = (z1_norm.T @ z2_norm) / n
    on_diag = (torch.diagonal(cc) - 1).pow(2).sum()
    off_diag = (cc - torch.diag(torch.diagonal(cc))).pow(2).sum()
    loss = on_diag + off_w * off_diag
    return loss, cc.detach()


def pca_2d(z: torch.Tensor) -> np.ndarray:
    z = z - z.mean(dim=0, keepdim=True)
    u, s, vh = torch.linalg.svd(z, full_matrices=False)
    proj = (z @ vh[:2].T).cpu().numpy()
    return proj


def run_training(use_barlow: bool, steps: int = 1500, snapshot_every: int = 50, seed: int = 0):
    device = get_device()
    encoder = Encoder().to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=3e-4)

    frames = gather_frames(num_episodes=200, seed=seed).to(device)
    print(f"frames: {tuple(frames.shape)}  barlow={use_barlow}")

    rng_torch = torch.Generator(device="cpu").manual_seed(seed)
    snapshots = []
    cc_snapshots = []
    for step in range(steps):
        idx = torch.randint(0, frames.shape[0], (256,), generator=rng_torch)
        x = frames[idx]
        x_cpu = x.cpu()
        v1 = augment(x_cpu, rng_torch).to(device)
        v2 = augment(x_cpu, rng_torch).to(device)

        z1 = encoder(v1)
        z2 = encoder(v2)

        if use_barlow:
            loss, cc = barlow_loss(z1, z2)
        else:
            loss = F.mse_loss(z1, z2)
            cc = None

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % snapshot_every == 0:
            with torch.no_grad():
                sample_idx = torch.randint(0, frames.shape[0], (256,), generator=rng_torch)
                z = encoder(frames[sample_idx])
                proj = pca_2d(z)
                std = z.std(dim=0).mean().item()
            snapshots.append({"step": step, "proj": proj, "z_std": std})
            if cc is not None:
                cc_snapshots.append(cc.cpu().numpy())
            else:
                cc_snapshots.append(None)
            print(f"step {step:>5d}  loss {loss.item():.4f}  z_std {std:.4f}")
    return snapshots, cc_snapshots


def render_pca_animation(snaps_naive, snaps_barlow, out_path="part2_jepa/outputs/pca_collapse_vs_barlow.gif"):
    frames = []
    all_proj = np.concatenate(
        [s["proj"] for s in snaps_naive] + [s["proj"] for s in snaps_barlow]
    )
    lim = float(np.abs(all_proj).max() * 1.05) + 1e-3

    img_size = 320
    margin = 30

    def draw_panel(proj: np.ndarray) -> np.ndarray:
        canvas = np.full((img_size, img_size), 255, dtype=np.uint8)
        if len(proj) == 0:
            return canvas
        xs = ((proj[:, 0] / lim + 1) / 2 * (img_size - 2 * margin) + margin).astype(int)
        ys = ((proj[:, 1] / lim + 1) / 2 * (img_size - 2 * margin) + margin).astype(int)
        for x, y in zip(xs, ys):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx * dx + dy * dy <= 4:
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < img_size and 0 <= yy < img_size:
                            canvas[yy, xx] = 30
        return canvas

    n = min(len(snaps_naive), len(snaps_barlow))
    for i in range(n):
        left = draw_panel(snaps_naive[i]["proj"])
        right = draw_panel(snaps_barlow[i]["proj"])
        sep = np.full((img_size, 8), 200, dtype=np.uint8)
        row = np.concatenate([left, sep, right], axis=1)
        frames.append(row)

    os.makedirs("part2_jepa/outputs", exist_ok=True)
    imageio.mimsave(out_path, frames, duration=0.08, loop=0)
    print(f"wrote {out_path}  ({len(frames)} frames)")


def colorize_diverging(arr: np.ndarray) -> np.ndarray:
    """Map [-1, 1] to a red-white-blue diverging colormap."""
    arr = np.clip(arr, -1, 1)
    pos = np.clip(arr, 0, 1)
    neg = np.clip(-arr, 0, 1)
    r = (255 * (1 - neg)).astype(np.uint8)
    g = (255 * (1 - pos - neg)).clip(0, 255).astype(np.uint8)
    b = (255 * (1 - pos)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def render_cross_corr(cc_snapshots, out_path="part2_jepa/outputs/cross_corr_evolution.png"):
    indices = [0, len(cc_snapshots) // 4, len(cc_snapshots) // 2, len(cc_snapshots) - 1]
    panels = []
    for i in indices:
        cc = cc_snapshots[i]
        if cc is None:
            continue
        rgb = colorize_diverging(cc)
        rgb = np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1)
        panels.append(rgb)
    if not panels:
        return
    sep = np.full((panels[0].shape[0], 8, 3), 230, dtype=np.uint8)
    cols = []
    for i, p in enumerate(panels):
        cols.append(p)
        if i < len(panels) - 1:
            cols.append(sep)
    strip = np.concatenate(cols, axis=1)
    imageio.imwrite(out_path, strip)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    print(">>> Naive joint-embedding (will collapse)")
    snaps_naive, _ = run_training(use_barlow=False, steps=1500, snapshot_every=50)
    print(">>> Barlow twins (decorrelated, no collapse)")
    snaps_barlow, cc_snaps = run_training(use_barlow=True, steps=1500, snapshot_every=50)

    render_pca_animation(snaps_naive, snaps_barlow)
    render_cross_corr(cc_snaps)

    final_naive_std = snaps_naive[-1]["z_std"]
    final_barlow_std = snaps_barlow[-1]["z_std"]
    print(f"\nfinal z_std  naive {final_naive_std:.5f}  barlow {final_barlow_std:.4f}")
