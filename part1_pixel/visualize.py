"""Render side-by-side comparison of pixel-space prediction vs JEPA prediction.

For a long bouncing-ball sequence, we predict the next frame at each step
using:
  1. The pixel-space baseline trained with MSE (blurry near walls)
  2. The JEPA encoder + predictor + viz decoder (crisp, commits to a direction)

Outputs:
  part1_pixel/outputs/comparison.gif  - animated comparison
  part1_pixel/outputs/comparison.png  - static frame-strip for the article
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch

from shared.data import simulate_episode, simulate_episode_noisy
from shared.encoder import Encoder, Decoder, get_device
from part1_pixel.train_pixel import PixelPredictor
from part2_jepa.train_jepa import Predictor


def load_pixel_model(device, noisy: bool = False):
    path = "checkpoints/pixel_noisy.pt" if noisy else "checkpoints/pixel.pt"
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = PixelPredictor().to(device)
    model.load_state_dict(ckpt["model"])
    return model


def load_jepa_models(device, noisy: bool = False):
    j_path = "checkpoints/jepa_noisy.pt" if noisy else "checkpoints/jepa.pt"
    d_path = "checkpoints/jepa_decoder_noisy.pt" if noisy else "checkpoints/jepa_decoder.pt"
    j = torch.load(j_path, map_location=device, weights_only=True)
    d = torch.load(d_path, map_location=device, weights_only=True)
    encoder = Encoder().to(device)
    encoder.load_state_dict(j["encoder"])
    predictor = Predictor().to(device)
    predictor.load_state_dict(j["predictor"])
    decoder = Decoder().to(device)
    decoder.load_state_dict(d["decoder"])
    return encoder, predictor, decoder


def upscale(img: np.ndarray, factor: int = 4) -> np.ndarray:
    return img.repeat(factor, axis=0).repeat(factor, axis=1)


def label_band(text: str, width: int, height: int = 18) -> np.ndarray:
    """Tiny text label rendered as a flat grey band. We don't render real text
    here to keep the dependency surface small — the article caption supplies
    labels. We just emit a uniform separator band."""
    return np.full((height, width), 0.6, dtype=np.float32)


def compose_row(frames: list[np.ndarray], gap: int = 8) -> np.ndarray:
    h = frames[0].shape[0]
    sep = np.full((h, gap), 0.5, dtype=np.float32)
    cols = []
    for i, f in enumerate(frames):
        cols.append(f)
        if i < len(frames) - 1:
            cols.append(sep)
    return np.concatenate(cols, axis=1)


def render_comparison(num_frames: int = 80, seed: int = 7, noisy: bool = False):
    device = get_device()
    print(f"device: {device}, noisy: {noisy}")

    pixel_model = load_pixel_model(device, noisy=noisy)
    encoder, predictor, decoder = load_jepa_models(device, noisy=noisy)

    rng = np.random.default_rng(seed)
    if noisy:
        episode = simulate_episode_noisy(steps=num_frames + 1, rng=rng)
    else:
        episode = simulate_episode(steps=num_frames + 1, rng=rng)  # (T+1, H, W)

    out_frames = []
    for t in range(num_frames):
        x_t = torch.from_numpy(episode[t]).to(device)[None, None, :, :]
        x_true = episode[t + 1]

        with torch.no_grad():
            x_pixel_pred = pixel_model(x_t).cpu().numpy()[0, 0]
            z_t = encoder(x_t)
            z_pred = predictor(z_t)
            x_jepa_pred = decoder(z_pred).cpu().numpy()[0, 0]

        row = compose_row(
            [
                upscale(episode[t]),
                upscale(x_true),
                upscale(x_pixel_pred),
                upscale(x_jepa_pred),
            ]
        )
        out_frames.append((row * 255).clip(0, 255).astype(np.uint8))

    os.makedirs("part1_pixel/outputs", exist_ok=True)
    suffix = "_noisy" if noisy else ""
    gif_path = f"part1_pixel/outputs/comparison{suffix}.gif"
    imageio.mimsave(gif_path, out_frames, duration=0.1, loop=0)
    print(f"wrote {gif_path}  ({len(out_frames)} frames)")

    # Pick a frame near the centre of the frame (most ambiguous future direction)
    # so the static strip shows the strongest smear in the pixel-prediction panel.
    centre_step = int(np.argmin(np.abs(episode[:num_frames].sum(axis=1).argmax(axis=1) - 32)))
    strip = out_frames[centre_step]
    png_path = f"part1_pixel/outputs/comparison{suffix}.png"
    imageio.imwrite(png_path, strip)
    print(f"wrote {png_path}  (frame {centre_step})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--noisy", action="store_true")
    args = parser.parse_args()
    render_comparison(num_frames=args.frames, noisy=args.noisy)
