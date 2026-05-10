# tiny-bouncing-jepa

Companion repo to a five-part JEPA blog series on danieljohnmorris.com. One synthetic bouncing-ball toy, four code parts (article Parts 2-5). Article Part 1 is argument-only with no code.

## Naming asymmetry to remember

The repo folder numbering is offset by one from the article numbering, because article Part 1 has no code:

| Article | Repo folder |
|---|---|
| Part 2 (smear) | `part1_pixel/` |
| Part 3 (VICReg + collapse) | `part2_jepa/` |
| Part 4 (DINOv2 hover) | `part3_production/` |
| Part 5 (planning) | `part4_planning/` |

When the user says "Part N" they mean the article; when a file is in `partN/` it means the code.

## Running

- All Python entrypoints assume working directory is the repo root and `PYTHONPATH=.` so that `from shared.data import ...` resolves.
- `.venv/` exists at the repo root.
- MPS is the target device (M-series Mac). No CUDA.

## What the toy can and can't show

The bouncing ball is a 64x64 grayscale frame with a single ball of radius 8. Direction info is not in any single frame.

**The toy demonstrates cleanly:**
- Pixel-MSE smear on bimodal futures (Part 2 / `part1_pixel/`)
- Representation collapse in naive joint-embedding training, and the Barlow Twins / VICReg fix (Part 3 / `part2_jepa/`)
- Action-conditioned prediction escaping the bimodal-averaging problem (Part 5 / `part4_planning/`)

**The toy cannot demonstrate:**
- JEPA producing a "sharper next-frame prediction" than pixel-MSE on single-frame input. Information-theoretically impossible. Don't try to render this; it failed many times during series drafting.
- JEPA's encoder discarding irrelevant detail in any visible way - the toy has no irrelevant detail to discard. Part 4 (`part3_production/`) uses DINOv2 on a real image for this demonstration instead.

## Standing methodological rules

- **The JEPA visualisation decoder is a hack.** It's not part of JEPA; it's an inverter we tack on to render embeddings as pixels. Whatever targets we train it on (clean vs noisy) biases the picture. The numerical loss comparisons are the honest measurements; the rendered panels are illustrations.
- **Don't manufacture an asymmetric demo to make JEPA look like it's winning.** If the toy can't show it, say so and let the article series carry the argument across parts.
- **The action-conditioned variant in `part4_planning/` is the only setup where the predictor's sim loss reaches `<0.01` on this toy.** That's the signal that the bimodal-future problem is resolved by architecture, not by tuning.

## Existing branches

- `master` - canonical, all five parts shipped
- `multi-frame` - experimental two-frame-input variant, kept for reference but not used by any article. The `master` Part 5 uses single-frame + action conditioning instead.

## Series outputs to keep stable

The dan-site article references these asset paths under `public/jepa/`:

- `comparison.png`, `comparison.gif` - 3-panel pixel-MSE smear (Part 2)
- `pca_collapse_vs_barlow.gif` - PCA scatter (Part 3)
- `cross_corr_evolution.png` - cross-correlation matrix evolution (Part 3)
- `hover_strip.png`, `hover_body.png`, `hover_face.png`, `hover_background.png` - DINOv2 hover demo (Part 4)
- `plan_strip.png` - planner ground-truth vs rollout strip (Part 5)

If any of these regenerate, copy them across to `/Users/dan/code/dan-site/public/jepa/` to keep the article visuals in sync.
