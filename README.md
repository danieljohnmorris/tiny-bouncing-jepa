# Tiny Bouncing JEPA

A from-scratch exploration of Yann LeCun's argument that LLMs are not enough and joint-embedding architectures with world models are. One synthetic bouncing ball, four code parts. Runs on an M-series Mac via MPS, no CUDA, under five minutes per part.

Companion repo to a five-part blog series on [danieljohnmorris.com](https://danieljohnmorris.com/writing). Article Part 1 is argument-only with no code; the four code folders below map to article Parts 2-5.

## Series

| Article | Repo folder | What it shows |
|---|---|---|
| [Part 1 - Yann LeCun's Bet Against LLMs](https://danieljohnmorris.com/writing/yann-lecun-bet-against-llms) | (no code) | The argument and papers behind the joint-embedding family. |
| [Part 2 - Why Pixel Prediction Goes Blurry](https://danieljohnmorris.com/writing/why-pixel-prediction-goes-blurry) | `part1_pixel/` | Pixel-space MSE next-frame predictor. The bimodal-future smear. |
| [Part 3 - Predict Embeddings, Not Pixels](https://danieljohnmorris.com/writing/predict-embeddings-not-pixels) | `part2_jepa/` | VICReg JEPA + Barlow Twins collapse demo. Cross-correlation matrix loss. |
| [Part 4 - From Representations to World Models](https://danieljohnmorris.com/writing/from-representations-to-world-models) | `part3_production/` | DINOv2 (DINOv3 if you have access) patch-similarity hover demo on a real image. |
| [Part 5 - Planning in Latent Space](https://danieljohnmorris.com/writing/planning-in-latent-space) | `part4_planning/` | Action-conditioned JEPA + brute-force MPC planner in latent space. |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy imageio transformers Pillow

# Part 2 - pixel baseline (~25s)
PYTHONPATH=. python part1_pixel/train_pixel.py
PYTHONPATH=. python part1_pixel/visualize.py

# Part 3 - JEPA + collapse demo (~3min total)
PYTHONPATH=. python part2_jepa/train_jepa.py
PYTHONPATH=. python part2_jepa/train_decoder.py
PYTHONPATH=. python part2_jepa/collapse_demo.py

# Part 4 - DINOv2 hover demo on a labrador (~20s, after model download)
PYTHONPATH=. python part3_production/hover_demo.py

# Part 5 - action-conditioned JEPA + MPC planner (~1min)
PYTHONPATH=. python part4_planning/train_action_jepa.py
PYTHONPATH=. python part4_planning/plan.py
```

All scripts run from the repo root with `PYTHONPATH=.` so the `shared/` and cross-folder imports resolve.

## Repo layout

```
shared/
  data.py            synthetic bouncing-ball dataset + noise variants + action triples
  encoder.py         tiny ConvNet encoder + decoder, get_device()

part1_pixel/
  train_pixel.py     pixel-space next-frame predictor (MSE)
  visualize.py       3-panel input/actual/prediction comparison
  outputs/           comparison.png, comparison.gif

part2_jepa/
  train_jepa.py      VICReg JEPA encoder + predictor (sim, var, cov losses)
  train_decoder.py   visualisation decoder for JEPA embeddings
  collapse_demo.py   naive joint-embedding collapse vs Barlow Twins fix
  outputs/           pca_collapse_vs_barlow.gif, cross_corr_evolution.png

part3_production/
  hover_demo.py      DINOv3/DINOv2 inference + per-patch cosine similarity heatmaps
  inputs/            labrador.jpg (auto-downloaded public domain image)
  outputs/           hover_strip.png, hover_{body,face,background}.png

part4_planning/
  train_action_jepa.py  action-conditioned predictor (z_t, action) -> z_{t+1}
  plan.py               enumeration-based MPC planner in latent space
  outputs/              plan_strip.png

checkpoints/         gitignored, written by training scripts
```

## Notes

- All models are tiny by design: 200K-1.1M parameters total. The point is testing the principle, not the model.
- All artifacts in `outputs/` are committed pre-rendered, so cloning gives you the visuals immediately.
- DINOv3 is gated on Hugging Face. `hover_demo.py` tries DINOv3 first, falls back to DINOv2-base. Both demonstrate the same architectural property.
- The single-frame toy is information-theoretically incapable of escaping bimodal-future averaging - this is honest in Parts 2/3 and resolved in Part 5 by adding direction as an explicit input to the predictor, matching V-JEPA 2-AC.
- Real production JEPAs (I-JEPA, V-JEPA 2, DINOv3, LeJEPA) use Vision Transformers and 300M-1.8B parameters. See [V-JEPA 2](https://github.com/facebookresearch/vjepa2) and [LeJEPA](https://github.com/rbalestr-lab/lejepa) for the production codebases.

## Numerical headline

JEPA sim loss across variants, all on the same bouncing-ball generator:

| Variant | Sim loss | Why |
|---|---|---|
| Single-frame, clean | 0.082 | Predictor averages bimodal next-embeddings |
| Single-frame, noisy distractors | 0.021 | Encoder drops noise, predictable signal concentrated |
| Multi-frame (2 frames), noisy | 0.032 | Direction implicit in two-frame input |
| **Action-conditioned** | **0.0057** | Direction given explicitly - bimodality gone |

Pixel-MSE loss for comparison: clean 0.018, noisy 0.060. Pixel prediction degrades 3-4x as irrelevant detail is added; JEPA does not.

## References

- LeCun, [A Path Towards Autonomous Machine Intelligence (2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- Zbontar, Jing, Misra, LeCun, Deny, [Barlow Twins (2021)](https://arxiv.org/abs/2103.03230)
- Bardes, Ponce, LeCun, [VICReg (2022)](https://arxiv.org/abs/2105.04906)
- Caron et al., [DINO (2021)](https://arxiv.org/abs/2104.14294)
- Assran et al., [I-JEPA (2023)](https://arxiv.org/abs/2301.08243)
- Meta FAIR, [V-JEPA 2 (2025)](https://arxiv.org/abs/2506.09985)
- Balestriero, LeCun, [LeJEPA (Nov 2025)](https://arxiv.org/abs/2511.08544)
- Welch Labs, [Yann LeCun's $1B Bet Against LLMs](https://www.youtube.com/watch?v=kYkIdXwW2AE)
