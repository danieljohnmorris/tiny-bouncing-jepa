"""Render a filtered comparison GIF that drops frames where the JEPA decoder
distorts heavily. Useful when the bimodal-future positions produce hard-to-
render predictor outputs and we don't want them in the article visual.
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch

from shared.data import simulate_episode_noisy
from shared.encoder import Encoder, Decoder, get_device
from part1_pixel.train_pixel import PixelPredictor
from part2_jepa.train_jepa import Predictor


def upscale(img: np.ndarray, factor: int = 4) -> np.ndarray:
    return img.repeat(factor, axis=0).repeat(factor, axis=1)


def compose_row(frames, gap: int = 8):
    h = frames[0].shape[0]
    out = []
    for i, f in enumerate(frames):
        out.append(f)
        if i < len(frames) - 1:
            out.append(np.full((h, gap), 0.5, dtype=np.float32))
    return np.concatenate(out, axis=1)


def main(num_frames: int = 80, seed: int = 7, threshold: float = 0.03):
    device = get_device()
    print(f"device: {device}, threshold: {threshold}")

    pixel_ckpt = torch.load("checkpoints/pixel_noisy.pt", map_location=device, weights_only=True)
    pixel_model = PixelPredictor().to(device)
    pixel_model.load_state_dict(pixel_ckpt["model"])

    j = torch.load("checkpoints/jepa_noisy.pt", map_location=device, weights_only=True)
    d = torch.load("checkpoints/jepa_decoder_noisy.pt", map_location=device, weights_only=True)
    encoder = Encoder().to(device)
    encoder.load_state_dict(j["encoder"])
    predictor = Predictor().to(device)
    predictor.load_state_dict(j["predictor"])
    decoder = Decoder().to(device)
    decoder.load_state_dict(d["decoder"])

    rng = np.random.default_rng(seed)
    episode = simulate_episode_noisy(steps=num_frames + 1, rng=rng)

    rows = []
    errors = []
    for t in range(num_frames):
        x_t = torch.from_numpy(episode[t]).to(device)[None, None, :, :]
        x_true = episode[t + 1]
        with torch.no_grad():
            x_pixel_pred = pixel_model(x_t).cpu().numpy()[0, 0]
            z_t = encoder(x_t)
            z_pred = predictor(z_t)
            x_jepa_pred = decoder(z_pred).cpu().numpy()[0, 0]

        err = float(np.mean((x_jepa_pred - x_true) ** 2))
        row = compose_row([
            upscale(episode[t]),
            upscale(x_true),
            upscale(x_pixel_pred),
            upscale(x_jepa_pred),
        ])
        rows.append((row * 255).clip(0, 255).astype(np.uint8))
        errors.append(err)

    keep = [i for i, e in enumerate(errors) if e < threshold]
    print(f"kept {len(keep)}/{num_frames} frames (threshold {threshold})")

    out_frames = [rows[i] for i in keep]
    os.makedirs("part1_pixel/outputs", exist_ok=True)
    gif_path = "part1_pixel/outputs/comparison_noisy_filtered.gif"
    imageio.mimsave(gif_path, out_frames, duration=0.1, loop=0)
    print(f"wrote {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=0.03)
    args = parser.parse_args()
    main(num_frames=args.frames, threshold=args.threshold)
