"""DINOv3 patch-similarity hover demo on a still image.

Loads DINOv3 (Meta, August 2025), passes an image through the encoder, and
extracts one embedding vector per 16x16 patch. For a chosen query patch,
computes cosine similarity to all other patches and renders a heatmap on
top of the original image. The result shows what the encoder considers
semantically similar to the query patch.

This is the production-grade joint-embedding encoder doing on real data
what Barlow Twins, VICReg, and LeJEPA are designed for. No fine-tuning,
no labels, just the frozen self-supervised encoder.

Falls back from DINOv3 to DINOv2 if DINOv3 is gated and not accessible
without auth.
"""

import argparse
import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DEFAULT_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
DEFAULT_IMAGE_PATH = "part3_production/inputs/labrador.jpg"

DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV2_MODEL = "facebook/dinov2-base"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fetch_image(path: str = DEFAULT_IMAGE_PATH, url: str = DEFAULT_IMAGE_URL) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {url} -> {path}")
        req = urllib.request.Request(url, headers={"User-Agent": "tiny-bouncing-jepa/0.1"})
        with urllib.request.urlopen(req) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def load_model(device, prefer_dinov3: bool = True):
    candidates = [DINOV3_MODEL, DINOV2_MODEL] if prefer_dinov3 else [DINOV2_MODEL]
    for name in candidates:
        try:
            print(f"trying {name}")
            processor = AutoImageProcessor.from_pretrained(name)
            model = AutoModel.from_pretrained(name).to(device).train(False)
            print(f"loaded {name}")
            return name, processor, model
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
    raise RuntimeError("could not load DINOv3 or DINOv2")


def get_patch_features(model, processor, image_path: str, device, target_size: int = 448):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)

    inputs = processor(images=img, return_tensors="pt", do_resize=False, do_center_crop=False)
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        out = model(pixel_values=pixel_values, interpolate_pos_encoding=True)

    # Skip CLS + register tokens; keep patch tokens.
    num_register = getattr(model.config, "num_register_tokens", 0)
    patch_tokens = out.last_hidden_state[:, 1 + num_register:, :]  # (1, N, D)

    patch_size = getattr(model.config, "patch_size", 16)
    grid = target_size // patch_size
    expected = grid * grid
    assert patch_tokens.shape[1] == expected, (
        f"expected {expected} patch tokens, got {patch_tokens.shape[1]} "
        f"(grid {grid}x{grid}, patch_size {patch_size})"
    )

    # L2-normalise so dot product = cosine similarity.
    patches = F.normalize(patch_tokens.squeeze(0), dim=-1)  # (N, D)
    return img, patches.cpu().numpy(), grid


def similarity_heatmap(patches: np.ndarray, query_idx: int) -> np.ndarray:
    """Cosine similarity from query patch to all patches. Returns (N,)."""
    return patches @ patches[query_idx]


def render_overlay(img: Image.Image, heatmap_grid: np.ndarray, grid: int,
                   query_xy: tuple[int, int], target_size: int = 448) -> np.ndarray:
    """Blend image with heatmap. Heatmap upsampled, marked with query crosshair."""
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H, W, 3)

    # Renormalise heatmap to [0, 1] for display.
    h = heatmap_grid.copy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)

    # Upsample to image size.
    h_up = np.kron(h, np.ones((target_size // grid, target_size // grid)))
    h_up = h_up[:target_size, :target_size]

    # Red-tint the heatmap, blend with image.
    cmap = np.stack([h_up, np.zeros_like(h_up), 1.0 - h_up], axis=-1)
    blended = 0.55 * arr + 0.45 * cmap

    # Crosshair on the query patch.
    qx, qy = query_xy
    px = qx * (target_size // grid)
    py = qy * (target_size // grid)
    s = target_size // grid
    blended[max(0, py - 1):py + s + 1, px:px + 2] = [1, 1, 1]
    blended[py:py + s, max(0, px - 1):px + 2] = [1, 1, 1]
    blended[py:py + s, px + s - 2:px + s + 1] = [1, 1, 1]
    blended[py + s - 2:py + s + 1, px:px + s] = [1, 1, 1]

    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def main(args):
    device = get_device()
    print(f"device: {device}")

    image_path = fetch_image()
    name, processor, model = load_model(device, prefer_dinov3=not args.dinov2)

    img, patches, grid = get_patch_features(model, processor, image_path, device,
                                             target_size=args.size)
    print(f"image: {img.size}, grid: {grid}x{grid}, patches: {patches.shape}")

    query_points = [
        ("body", grid // 2, int(grid * 0.55)),
        ("face", int(grid * 0.42), int(grid * 0.32)),
        ("background", int(grid * 0.85), int(grid * 0.15)),
    ]

    os.makedirs("part3_production/outputs", exist_ok=True)
    panels = []
    for label, qx, qy in query_points:
        idx = qy * grid + qx
        heatmap = similarity_heatmap(patches, idx).reshape(grid, grid)
        overlay = render_overlay(img, heatmap, grid, (qx, qy), target_size=args.size)
        panels.append(overlay)
        from PIL import Image as PImage
        PImage.fromarray(overlay).save(f"part3_production/outputs/hover_{label}.png")
        print(f"wrote part3_production/outputs/hover_{label}.png")

    # Combined strip.
    sep = np.full((args.size, 8, 3), 80, dtype=np.uint8)
    cols = []
    for i, p in enumerate(panels):
        cols.append(p)
        if i < len(panels) - 1:
            cols.append(sep)
    strip = np.concatenate(cols, axis=1)
    from PIL import Image as PImage
    PImage.fromarray(strip).save("part3_production/outputs/hover_strip.png")
    print(f"wrote part3_production/outputs/hover_strip.png")
    print(f"\nused model: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=448, help="image size (must divide patch size cleanly)")
    parser.add_argument("--dinov2", action="store_true", help="skip DINOv3 attempt, go straight to DINOv2")
    args = parser.parse_args()
    main(args)
