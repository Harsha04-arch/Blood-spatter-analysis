"""
Module 4 — 2D Area of Convergence calculation.

Algorithm:
  For each eligible elongated droplet:
    1. Extract centroid (x, y) and major-axis orientation θ.
    2. Construct an infinite trajectory line through the centroid.
    3. Compute all pairwise line intersections.
    4. Estimate convergence point via robust least-squares / weighted mean.
    5. Compute convergence error (RMS distance of intersection cloud).

The convergence point is derived entirely from droplet geometry —
it is NOT forced toward the image center.
"""

import numpy as np
from typing import Optional
from backend.utils.logger import get_logger

log = get_logger(__name__)

# Minimum aspect ratio to consider a stain "elongated" for trajectory use
MIN_ASPECT_RATIO = 1.01
# Minimum number of droplets required
MIN_DROPLETS = 2


def compute_2d_convergence(droplets: list[dict]) -> dict:
    """
    Compute 2D area of convergence from detected droplets.

    Args:
        droplets: List of droplet dicts (from droplet_detection).

    Returns:
        dict with convergence_x, convergence_y, error_px, intersections, used_droplets.
    """
    # Filter elongated stains with valid angles
    eligible = [
        d for d in droplets
        if d.get("aspect_ratio", 1.0) >= MIN_ASPECT_RATIO
        and d.get("angle_valid", False)
        and d.get("major_axis", 0) > 0
    ]

    if len(eligible) < MIN_DROPLETS:
        return {
            "status": "error",
            "message": (
                f"Too few elongated droplets for convergence analysis. "
                f"Found {len(eligible)}, need at least {MIN_DROPLETS}. "
                "Ensure the image contains elongated bloodstains."
            ),
            "used_droplets": len(eligible),
        }

    # Build line parameters: (cx, cy, angle_rad)
    lines = []
    for d in eligible:
        cx  = d["cx"]
        cy  = d["cy"]
        # orientation from ellipse fitting is the angle of the major axis
        ang_deg = d.get("orientation", 0.0)
        ang_rad = np.deg2rad(ang_deg)
        # Direction vector along major axis
        dx = np.cos(ang_rad)
        dy = np.sin(ang_rad)
        lines.append((cx, cy, dx, dy, d["id"]))

    # Pairwise intersections
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                intersections.append(pt)

    if len(intersections) == 0:
        return {
            "status": "error",
            "message": "No trajectory intersections found (all lines may be parallel).",
            "used_droplets": len(eligible),
        }

    pts = np.array(intersections)  # (N, 2)

    # Robust convergence: remove outliers > 2σ, then compute weighted mean
    pts_filtered, weights = _ransac_filter(pts)

    conv_x = float(np.average(pts_filtered[:, 0], weights=weights))
    conv_y = float(np.average(pts_filtered[:, 1], weights=weights))

    # RMS error
    dists = np.sqrt((pts_filtered[:, 0] - conv_x)**2 + (pts_filtered[:, 1] - conv_y)**2)
    rms_error = float(np.sqrt(np.mean(dists**2)))

    log.info(f"2D convergence: ({conv_x:.1f}, {conv_y:.1f}), RMS={rms_error:.1f}px, "
             f"intersections={len(pts_filtered)}")

    return {
        "status": "success",
        "convergence_x":  round(conv_x, 2),
        "convergence_y":  round(conv_y, 2),
        "error_px":       round(rms_error, 2),
        "n_intersections": len(pts_filtered),
        "n_outliers":     len(pts) - len(pts_filtered),
        "intersections":  [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in pts_filtered],
        "used_droplets":  len(eligible),
        "eligible_ids":   [l[4] for l in lines],
        "label":          "Geometric Estimation",
        "disclaimer": (
            "2D convergence is a geometric estimate based on image-plane stain orientations. "
            "Accuracy depends on image quality, surface properties, and perspective."
        ),
    }


def _line_intersection(
    l1: tuple, l2: tuple
) -> Optional[tuple[float, float]]:
    """
    Compute intersection of two parametric lines.
    l = (cx, cy, dx, dy, id)
    Returns (x, y) or None if lines are parallel.
    """
    x1, y1, d1x, d1y, _ = l1
    x2, y2, d2x, d2y, _ = l2

    # Solve: (x1 + t*d1x, y1 + t*d1y) = (x2 + s*d2x, y2 + s*d2y)
    # d1x*t - d2x*s = x2 - x1
    # d1y*t - d2y*s = y2 - y1
    denom = d1x * (-d2y) - (-d2x) * d1y
    if abs(denom) < 1e-9:
        return None  # Parallel

    dx = x2 - x1
    dy = y2 - y1
    t = (dx * (-d2y) - (-d2x) * dy) / denom

    ix = x1 + t * d1x
    iy = y1 + t * d1y
    return (ix, iy)


def _ransac_filter(pts: np.ndarray, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Remove intersections farther than sigma*std from the centroid."""
    if len(pts) < 3:
        return pts, np.ones(len(pts))

    center = np.median(pts, axis=0)
    dists  = np.sqrt(((pts - center)**2).sum(axis=1))
    thresh = np.mean(dists) + sigma * np.std(dists)
    mask   = dists <= thresh
    pts_f  = pts[mask]
    if len(pts_f) == 0:
        pts_f = pts
        mask  = np.ones(len(pts), dtype=bool)

    # Inverse-distance weights
    dists_f = np.sqrt(((pts_f - center)**2).sum(axis=1))
    weights = 1.0 / (dists_f + 1e-6)
    return pts_f, weights
