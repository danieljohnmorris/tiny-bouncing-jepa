# Tiny Bouncing JEPA

A from-scratch exploration of Yann LeCun's argument that LLMs are not enough and joint-embedding architectures with world models are. Four parts. One synthetic bouncing ball. Runs on an M-series Mac via MPS, no CUDA, under five minutes total per part.

This is the companion repo to a four-part blog series on [danieljohnmorris.com](https://danieljohnmorris.com/writing). Each folder maps to one post.

## Series

- **Part 1 - Why Pixel Prediction Goes Blurry** (`part1_pixel/`) - bouncing-ball dataset, MSE next-frame predictor, the smear that LeCun warns about.
- **Part 2 - Predict Embeddings, Not Pixels** (`part2_jepa/`) - VICReg JEPA, Barlow-Twins collapse demo, LeJEPA's SIGReg cleanup.
- **Part 3 - From Representations to World Models** (`part3_production/`) - DINOv3 patch-similarity hover demo on a still image.
- **Part 4 - Planning in Latent Space** (`part4_planning/`) - action-conditioned JEPA, MPC-style planning in latent space.

Parts 3 and 4 are stubs at the time of first commit.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy imageio

# Part 1 - pixel baseline (~25s)
python -m part1_pixel.train_pixel

# Part 2 - JEPA with VICReg (~3min), viz decoder, collapse demo
python -m part2_jepa.train_jepa
python -m part2_jepa.train_decoder
python -m part2_jepa.collapse_demo

# Render the side-by-side comparison from Part 1
python -m part1_pixel.visualize
```

All scripts assume the working directory is the repo root. Run with `python -m <package>.<module>`.

## Repo layout

```
shared/
  data.py          synthetic bouncing-ball dataset
  encoder.py       tiny ConvNet encoder + decoder, get_device()

part1_pixel/
  train_pixel.py   pixel-space next-frame predictor (MSE)
  visualize.py     side-by-side rendering
  outputs/         comparison.png, comparison.gif

part2_jepa/
  train_jepa.py    VICReg JEPA
  train_decoder.py viz decoder for JEPA embeddings
  collapse_demo.py representation collapse + Barlow Twins fix
  outputs/         pca_collapse_vs_barlow.gif, cross_corr_evolution.png

part3_production/  (Part 3 - DINOv3 hover demo, stub)
part4_planning/    (Part 4 - latent-space planner, stub)

checkpoints/       gitignored, written by the training scripts
```

## Notes

- All models are tiny by design: 700K to 1.1M parameters total. The point is testing the principle, not the model.
- All artifacts in `outputs/` directories are checked in, pre-rendered. Cloning the repo gives you the visuals immediately.
- Real production JEPAs (I-JEPA, V-JEPA 2, DINOv3, LeJEPA) use Vision Transformers and 300M-1.8B parameters. See [V-JEPA 2](https://github.com/facebookresearch/vjepa2) and [LeJEPA](https://github.com/rbalestr-lab/lejepa) for the production codebases.

## References

- LeCun, [A Path Towards Autonomous Machine Intelligence (2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- Zbontar, Jing, Misra, LeCun, Deny, [Barlow Twins (2021)](https://arxiv.org/abs/2103.03230)
- Bardes, Ponce, LeCun, [VICReg (2022)](https://arxiv.org/abs/2105.04906)
- Assran et al., [I-JEPA (2023)](https://arxiv.org/abs/2301.08243)
- Meta FAIR, [V-JEPA 2 (2025)](https://arxiv.org/abs/2506.09985)
- Balestriero, LeCun, [LeJEPA (Nov 2025)](https://arxiv.org/abs/2511.08544)
- Welch Labs, [Yann LeCun's $1B Bet Against LLMs](https://www.youtube.com/watch?v=kYkIdXwW2AE)
