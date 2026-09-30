"""
Module 4 — 3D Point of Origin Resolver.

Mathematical model:
  For each droplet with valid impact angle θ and position (x, y) in 2D,
  we construct a 3D ray emanating from the stain position.

  The ray direction is determined by:
    - Horizontal angle: derived from droplet orientation in the image plane
    - Vertical angle:   90° - θ  (where θ is the impact angle from horizontal)

  The ray equation:
    P(t) = origin + t * direction

  where:
    origin    = (cx, cy, 0)     [stain on z=0 plane]
    direction = (cos(φ)*cos(α), sin(φ)*cos(α), sin(α))
    φ = horizontal angle (from orientation)
    α = vertical angle = π/2 - impact_angle

  The point of origin is the 3D point that minimises the sum of
  squared distances to all trajectory rays (least-squares ray intersection).

  Reference: "Reconstruction of bloodstain origin from spatial positions
  and flight directions" — geometric approach.
"""

import numpy as np
from typing import Optional
from backend.utils.logger import get_logger

log = get_logger(__name__)

MIN_RAYS = 2


def compute_3d_origin(
    droplets: list[dict],
    convergence_x: float,
    convergence_y: float,
    image_scale_mm_per_px: float = 1.0,
) -> dict:
    """
    Estimate the 3D point of origin from droplet positions and impact angles.

    Args:
        droplets:             List of droplet dicts with impact_angle and orientation.
        convergence_x/y:      2D convergence point (used to resolve trajectory direction).
        image_scale_mm_per_px: Physical scale (default 1.0 = pixel units).

    Returns:
        dict with x, y, z, error, rays used.
    """
    rays = _build_rays(droplets, convergence_x, convergence_y, image_scale_mm_per_px)

    if len(rays) < MIN_RAYS:
        return {
            "status": "error",
            "message": (
                f"Insufficient valid rays for 3D point of origin estimation. "
                f"Found {len(rays)}, need at least {MIN_RAYS}. "
                "More elongated droplets with valid impact angles are required."
            ),
            "rays_used": len(rays),
        }

    origin_pt, rms_error = _least_squares_ray_intersection(rays)

    if origin_pt is None:
        return {
            "status": "error",
            "message": "3D ray intersection solver failed — rays may be nearly parallel.",
            "rays_used": len(rays),
        }

    x, y, z = origin_pt

    # Rays for visualization
    ray_data = [
        {
            "origin":    [round(float(r[0][0]), 2), round(float(r[0][1]), 2), round(float(r[0][2]), 2)],
            "direction": [round(float(r[1][0]), 4), round(float(r[1][1]), 4), round(float(r[1][2]), 4)],
            "droplet_id": r[2],
        }
        for r in rays
    ]

    return {
        "status": "success",
        "x": round(float(x), 2),
        "y": round(float(y), 2),
        "z": round(float(z), 2),
        "error_units": round(float(rms_error), 2),
        "rays_used": len(rays),
        "rays": ray_data,
        "label": "Estimated 3D Point of Origin",
        "disclaimer": (
            "This is a geometric estimate based on 2D image measurements. "
            "Accuracy is limited by image perspective, surface angle, and calibration. "
            "Results require validation by qualified forensic professionals."
        ),
        "units": "pixels (set image_scale_mm_per_px for real-world units)",
    }


def _build_rays(
    droplets: list[dict],
    conv_x: float,
    conv_y: float,
    scale: float,
) -> list[tuple]:
    """
    Build 3D rays from droplet measurements.

    Returns list of (origin_3d, direction_3d, droplet_id).
    """
    rays = []
    for d in droplets:
        if not d.get("angle_valid", False):
            continue
        impact_angle = d.get("impact_angle")
        if impact_angle is None:
            continue

        cx = d["cx"] * scale
        cy = d["cy"] * scale
        orient_deg = d.get("orientation", 0.0)

        # Horizontal direction: from stain toward convergence point
        dx_2d = conv_x * scale - cx
        dy_2d = conv_y * scale - cy
        horiz_mag = np.sqrt(dx_2d**2 + dy_2d**2)

        if horiz_mag < 1e-6:
            continue

        # Normalise horizontal component
        ux = dx_2d / horiz_mag
        uy = dy_2d / horiz_mag

        # Vertical component: tan(impact_angle) = horizontal / vertical
        # vertical rise per unit horizontal travel
        impact_rad = np.deg2rad(impact_angle)
        if abs(np.sin(impact_rad)) < 1e-9:
            continue  # grazing impact — undefined Z

        # Direction vector in 3D (stain plane is z=0, origin is above)
        # The blood fell from height z to point (cx, cy, 0) at angle θ
        # tan(θ) = horizontal_distance / height
        # So: direction = (ux*cos(θ), uy*cos(θ), sin(θ))
        cos_a = np.cos(impact_rad)
        sin_a = np.sin(impact_rad)

        direction = np.array([ux * cos_a, uy * cos_a, sin_a], dtype=float)
        origin    = np.array([cx, cy, 0.0], dtype=float)

        rays.append((origin, direction, d["id"]))

    return rays


def _least_squares_ray_intersection(rays: list[tuple]) -> tuple[Optional[np.ndarray], float]:
    """
    Least-squares estimate of the point minimizing squared distances to all rays.

    For a ray: P(t) = o + t*d
    The distance from point X to ray is ||(X - o) - ((X-o)·d)d||

    Solve: (I - sum(d*dᵀ)) * X = sum((I - d*dᵀ) * o)
    Reference: "Intersection of N lines in 3D — least-squares" (Kanatani, 1985)
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for (o, d, _) in rays:
        d = d / (np.linalg.norm(d) + 1e-12)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o

    try:
        X = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, float("inf")

    # RMS distance
    dists = []
    for (o, d, _) in rays:
        d = d / (np.linalg.norm(d) + 1e-12)
        diff = X - o
        proj = np.dot(diff, d)
        closest = o + proj * d
        dists.append(np.linalg.norm(X - closest))

    rms = float(np.sqrt(np.mean(np.array(dists)**2)))
    return X, rms
