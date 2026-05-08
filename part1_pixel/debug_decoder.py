"""Diagnose JEPA decoder distortion in the held-out visualisation episode.

For each frame in the GIF, score how well the decoder reconstruction matches
the actual next frame. Print the worst frames with their ball position so we
can see if the distortion clusters at specific positions.
"""

import numpy as np
import torch

from shared.data import simulate_episode_noisy, FRAME_SIZE
from shared.encoder import Encoder, Decoder, get_device
from part2_jepa.train_jepa import Predictor


def main():
    device = get_device()
    print(f"device: {device}")

    j = torch.load("checkpoints/jepa_noisy.pt", map_location=device, weights_only=True)
    d = torch.load("checkpoints/jepa_decoder_noisy.pt", map_location=device, weights_only=True)
    encoder = Encoder().to(device)
    encoder.load_state_dict(j["encoder"])
    predictor = Predictor().to(device)
    predictor.load_state_dict(j["predictor"])
    decoder = Decoder().to(device)
    decoder.load_state_dict(d["decoder"])

    rng = np.random.default_rng(7)
    episode = simulate_episode_noisy(steps=81, rng=rng)

    errors = []
    positions = []
    for t in range(80):
        x_t = torch.from_numpy(episode[t]).to(device)[None, None, :, :]
        with torch.no_grad():
            z_t = encoder(x_t)
            z_pred = predictor(z_t)
            x_jepa = decoder(z_pred).cpu().numpy()[0, 0]

        col_mass = episode[t].sum(axis=0)
        x = float((col_mass * np.arange(FRAME_SIZE)).sum() / col_mass.sum())

        diff = float(np.mean((x_jepa - episode[t + 1]) ** 2))
        errors.append(diff)
        positions.append(x)

    errors = np.array(errors)
    positions = np.array(positions)

    print(f"\nframe-level JEPA decoder MSE vs actual next frame:")
    print(f"  min  {errors.min():.4f}  max  {errors.max():.4f}  mean  {errors.mean():.4f}")

    print(f"\n10 worst frames:")
    worst = np.argsort(errors)[::-1][:10]
    for t in worst:
        near_wall = "wall" if (positions[t] < 16 or positions[t] > 48) else "    "
        print(f"  frame {t:>3d}  pos {positions[t]:>5.1f}  {near_wall}  err {errors[t]:.4f}")

    print(f"\n10 cleanest frames:")
    best = np.argsort(errors)[:10]
    for t in best:
        near_wall = "wall" if (positions[t] < 16 or positions[t] > 48) else "    "
        print(f"  frame {t:>3d}  pos {positions[t]:>5.1f}  {near_wall}  err {errors[t]:.4f}")


if __name__ == "__main__":
    main()
