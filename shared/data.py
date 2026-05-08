"""Synthetic bouncing-ball dataset.

A 1D ball moves left-right inside a 64x64 frame. Bouncing is deterministic
(reverses at walls). The ambiguity comes from the input format: each
training pair is (frame_t, frame_t+1) using only one input frame, which
contains no velocity information. The same ball position appears in many
trajectories moving leftward and rightward, so the next-frame distribution
is bimodal. A pixel-space MSE predictor is forced to predict the average
of the two modes — a blurry smear with two faint ghosts.
"""

import numpy as np
import torch


FRAME_SIZE = 64
BALL_RADIUS = 8
SPEED = 6
BG = 0.0
FG = 1.0


def render_frame(x: float, y: float) -> np.ndarray:
    frame = np.full((FRAME_SIZE, FRAME_SIZE), BG, dtype=np.float32)
    yy, xx = np.ogrid[:FRAME_SIZE, :FRAME_SIZE]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= BALL_RADIUS ** 2
    frame[mask] = FG
    return frame


def simulate_episode(steps: int, rng: np.random.Generator) -> np.ndarray:
    """Return (steps, H, W) array. Deterministic bouncing physics.

    Random initial position and direction, so across episodes the same
    position appears in trajectories travelling both ways.
    """
    x = rng.uniform(BALL_RADIUS + 1, FRAME_SIZE - BALL_RADIUS - 1)
    y = FRAME_SIZE // 2
    vx = float(rng.choice([-SPEED, SPEED]))
    frames = np.empty((steps, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    for t in range(steps):
        frames[t] = render_frame(x, y)
        x_next = x + vx
        if x_next - BALL_RADIUS < 0:
            vx = abs(vx)
            x_next = x + vx
        elif x_next + BALL_RADIUS >= FRAME_SIZE:
            vx = -abs(vx)
            x_next = x + vx
        x = x_next
    return frames


def make_pairs(num_pairs: int, seed: int = 0) -> torch.Tensor:
    """Return tensor of shape (num_pairs, 2, 1, H, W) of (frame_t, frame_t+1) pairs."""
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < num_pairs:
        episode = simulate_episode(steps=24, rng=rng)
        for t in range(len(episode) - 1):
            pairs.append(np.stack([episode[t], episode[t + 1]]))
            if len(pairs) >= num_pairs:
                break
    arr = np.stack(pairs)[:, :, None, :, :]
    return torch.from_numpy(arr)


NOISE_DOTS_PER_FRAME = 200


def add_distractor_noise(frame: np.ndarray, rng: np.random.Generator,
                         num_dots: int = NOISE_DOTS_PER_FRAME) -> np.ndarray:
    """Sprinkle a fresh batch of bright dots on the background of a single frame.

    The dots are i.i.d. across frames (no temporal structure), so they carry
    no predictable signal. A pixel-MSE next-frame predictor is forced to
    average them - producing a grey haze. A JEPA encoder has training
    pressure to drop them - the similarity loss rewards predictable
    embeddings, and embedding the noise makes prediction impossible.
    """
    out = frame.copy()
    ys = rng.integers(0, FRAME_SIZE, size=num_dots)
    xs = rng.integers(0, FRAME_SIZE, size=num_dots)
    out[ys, xs] = FG
    return out


def make_pairs_noisy(num_pairs: int, seed: int = 0,
                     num_dots: int = NOISE_DOTS_PER_FRAME) -> torch.Tensor:
    """Same as make_pairs, but each frame has independent distractor noise."""
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < num_pairs:
        episode = simulate_episode(steps=24, rng=rng)
        for t in range(len(episode) - 1):
            f_t = add_distractor_noise(episode[t], rng, num_dots=num_dots)
            f_next = add_distractor_noise(episode[t + 1], rng, num_dots=num_dots)
            pairs.append(np.stack([f_t, f_next]))
            if len(pairs) >= num_pairs:
                break
    arr = np.stack(pairs)[:, :, None, :, :]
    return torch.from_numpy(arr)


def simulate_episode_noisy(steps: int, rng: np.random.Generator,
                           num_dots: int = NOISE_DOTS_PER_FRAME) -> np.ndarray:
    """Bouncing-ball episode with i.i.d. distractor dots overlaid each frame."""
    clean = simulate_episode(steps, rng)
    out = np.empty_like(clean)
    for t in range(steps):
        out[t] = add_distractor_noise(clean[t], rng, num_dots=num_dots)
    return out


def make_pairs_noisy_input_clean_target(num_pairs: int, seed: int = 0,
                                         num_dots: int = NOISE_DOTS_PER_FRAME) -> torch.Tensor:
    """Pairs where frame_t has distractor noise but frame_{t+1} is clean.

    Used to train the visualisation decoder: the encoder/predictor see noisy
    inputs (so the encoder still has pressure to drop noise) but the decoder
    reconstructs the clean next frame (so panel 4 shows what JEPA's predicted
    embedding corresponds to in clean pixel space, not a hallucination of
    new noise patterns).
    """
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < num_pairs:
        episode = simulate_episode(steps=24, rng=rng)
        for t in range(len(episode) - 1):
            f_t = add_distractor_noise(episode[t], rng, num_dots=num_dots)
            f_next = episode[t + 1]  # clean target
            pairs.append(np.stack([f_t, f_next]))
            if len(pairs) >= num_pairs:
                break
    arr = np.stack(pairs)[:, :, None, :, :]
    return torch.from_numpy(arr)


if __name__ == "__main__":
    pairs = make_pairs(8)
    print("pairs shape:", pairs.shape)
    print("min/max:", pairs.min().item(), pairs.max().item())
    print("ball pixels per frame:", int((pairs[0, 0] > 0).sum().item()))
