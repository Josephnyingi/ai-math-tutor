"""
Visual grounding: count objects in a rendered image.

Two backends (selected automatically by available packages):
  1. BlobCounter  — fast, dependency-free baseline using scipy/skimage
  2. OWLViT-tiny  — open-vocabulary detector for richer category recognition

The active backend is chosen once at import time; the public API is identical
for both so the rest of the system is backend-agnostic.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------
# Blob counter baseline (no heavy deps)
# ------------------------------------------------------------------

def _blob_count(image: np.ndarray, min_area: int = 50) -> int:
    """
    Count distinct bright blobs in a grayscale or RGB image.
    Works well for simple rendered stimuli (solid objects on plain background).
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2]).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    # Threshold (Otsu-style: mean-based)
    thresh = gray.mean()
    binary = (gray > thresh).astype(np.uint8)

    # Connected components via simple flood-fill BFS
    visited = np.zeros_like(binary, dtype=bool)
    count = 0
    rows, cols = binary.shape

    def bfs(r0, c0):
        area = 0
        stack = [(r0, c0)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r, c] or binary[r, c] == 0:
                continue
            visited[r, c] = True
            area += 1
            stack.extend([(r+1,c),(r-1,c),(r,c+1),(r,c-1)])
        return area

    for r in range(rows):
        for c in range(cols):
            if binary[r, c] == 1 and not visited[r, c]:
                area = bfs(r, c)
                if area >= min_area:
                    count += 1
    return count


# ------------------------------------------------------------------
# OWLViT-tiny backend
# ------------------------------------------------------------------

_owlvit_pipeline = None

def _load_owlvit():
    global _owlvit_pipeline
    if _owlvit_pipeline is None:
        from transformers import pipeline
        _owlvit_pipeline = pipeline(
            "zero-shot-object-detection",
            model="google/owlvit-base-patch32",
            device=-1,  # CPU
        )
    return _owlvit_pipeline


def _owlvit_count(
    image: np.ndarray,
    query: str,
    score_threshold: float = 0.10,
) -> int:
    from PIL import Image as PILImage
    pipe = _load_owlvit()
    pil_img = PILImage.fromarray(image)
    results = pipe(pil_img, candidate_labels=[query], threshold=score_threshold)
    return len(results)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def count_objects(
    image: np.ndarray,
    query: str = "",
    backend: str = "auto",
) -> Tuple[int, str]:
    """
    Count objects in *image*.

    Args:
        image:   uint8 numpy array (H, W) or (H, W, 3)
        query:   text description of what to count (used by OWLViT only)
        backend: "blob" | "owlvit" | "auto"
                 "auto" tries OWLViT first; falls back to blob if unavailable

    Returns:
        (count, backend_used)
    """
    if backend == "blob":
        return _blob_count(image), "blob"

    if backend == "owlvit" or backend == "auto":
        try:
            n = _owlvit_count(image, query or "object")
            return n, "owlvit"
        except Exception:
            pass  # fall back

    return _blob_count(image), "blob"


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file to a uint8 numpy array."""
    from PIL import Image as PILImage
    img = PILImage.open(str(path)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def render_counting_stimulus(
    n: int,
    label: str = "●",
    grid_size: int = 128,
) -> np.ndarray:
    """
    Render a simple counting stimulus: *n* circles on a white background.
    Used when no pre-rendered image asset is available.
    Returns a (grid_size, grid_size, 3) uint8 array.
    """
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont
        img = PILImage.new("RGB", (grid_size, grid_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        margin = 10
        cols = min(n, 5)
        rows = (n + cols - 1) // cols
        cell_w = (grid_size - 2 * margin) // max(cols, 1)
        cell_h = (grid_size - 2 * margin) // max(rows, 1)
        r = min(cell_w, cell_h) // 3
        for i in range(n):
            col = i % cols
            row = i // cols
            cx = margin + col * cell_w + cell_w // 2
            cy = margin + row * cell_h + cell_h // 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 120, 220))
        return np.array(img, dtype=np.uint8)
    except ImportError:
        # Fallback: return blank array
        arr = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 240
        return arr
