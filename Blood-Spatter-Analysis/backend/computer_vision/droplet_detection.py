"""
Module 3 — Droplet detection using OpenCV contour analysis.

Pipeline:
  preprocessed binary image
  → connected component / contour analysis
  → ellipse fitting
  → per-droplet measurements
  → annotated output image
"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
from backend.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Droplet:
    """All measurements for a single detected droplet."""
    id: int
    cx: float          # centroid X
    cy: float          # centroid Y
    width: float       # bounding box width  (minor axis proxy)
    height: float      # bounding box height
    area: float
    perimeter: float
    aspect_ratio: float
    orientation: float # degrees from horizontal
    major_axis: float
    minor_axis: float
    equiv_diameter: float
    impact_angle: Optional[float]  # degrees, None if invalid
    angle_valid: bool
    angle_warning: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


def detect_droplets(
    morph: np.ndarray,
    original: np.ndarray,
    min_area: int = 50,
    max_area: int = 50000,
) -> tuple[list[Droplet], np.ndarray]:
    """
    Detect and measure bloodstain droplets from a preprocessed binary image.

    Args:
        morph:    Binary (morphologically processed) image, uint8.
        original: Original BGR image for annotation.
        min_area: Minimum droplet area in pixels.
        max_area: Maximum droplet area in pixels.

    Returns:
        (droplets, annotated_bgr)
    """
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    log.info(f"Raw contours found: {len(contours)}")

    droplets: list[Droplet] = []
    annotated = original.copy()

    drop_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Bounding rect
        x, y, w, h = cv2.boundingRect(cnt)

        # Moments → centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Perimeter
        perimeter = cv2.arcLength(cnt, True)

        # Aspect ratio (safe)
        aspect_ratio = round(max(w, h) / max(min(w, h), 1), 4)

        # Equivalent diameter
        equiv_diam = round(float(np.sqrt(4 * area / np.pi)), 2)

        # Ellipse fitting (needs ≥5 points)
        major_axis = float(max(w, h))
        minor_axis = float(min(w, h))
        orientation = 0.0

        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                (ex, ey), (e_minor, e_major), e_angle = ellipse
                if e_major > 0 and e_minor > 0:
                    major_axis  = round(float(max(e_major, e_minor)), 2)
                    minor_axis  = round(float(min(e_major, e_minor)), 2)
                    orientation = round(float(e_angle), 2)
                    # Draw ellipse on annotated image
                    cv2.ellipse(annotated, ellipse, (0, 255, 100), 1)
            except cv2.error:
                pass

        # Impact angle
        impact_angle, angle_valid, angle_warning = _compute_impact_angle(minor_axis, major_axis)

        droplet = Droplet(
            id=drop_id,
            cx=round(cx, 2),
            cy=round(cy, 2),
            width=round(float(w), 2),
            height=round(float(h), 2),
            area=round(float(area), 2),
            perimeter=round(float(perimeter), 2),
            aspect_ratio=aspect_ratio,
            orientation=orientation,
            major_axis=major_axis,
            minor_axis=minor_axis,
            equiv_diameter=equiv_diam,
            impact_angle=impact_angle,
            angle_valid=angle_valid,
            angle_warning=angle_warning,
        )
        droplets.append(droplet)

        # Annotate
        _annotate_droplet(annotated, droplet)
        drop_id += 1

    log.info(f"Valid droplets detected: {len(droplets)}")
    return droplets, annotated


def _compute_impact_angle(minor: float, major: float) -> tuple[Optional[float], bool, Optional[str]]:
    """
    Impact angle θ = arcsin(W / L) where W=minor axis, L=major axis.

    Returns (angle_degrees, is_valid, warning_message).
    """
    if major <= 0:
        return None, False, "Major axis is zero — cannot compute angle."
    ratio = minor / major
    if ratio < 0.0 or ratio > 1.0:
        return None, False, f"W/L ratio {ratio:.4f} out of valid range [0,1] — segmentation error."
    # Clamp floating-point precision issues
    ratio = max(0.0, min(1.0, ratio))
    angle = float(np.degrees(np.arcsin(ratio)))
    return round(angle, 2), True, None


def _annotate_droplet(img: np.ndarray, d: Droplet):
    """Draw centroid, ID label, and contour info on annotated image."""
    cx_i, cy_i = int(d.cx), int(d.cy)

    # Centroid dot
    cv2.circle(img, (cx_i, cy_i), 3, (0, 0, 255), -1)

    # Label
    angle_str = f"{d.impact_angle:.1f}°" if d.angle_valid and d.impact_angle is not None else "N/A"
    label = f"#{d.id} {angle_str}"

    font_scale = 0.35
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    tx = min(cx_i + 5, img.shape[1] - tw - 2)
    ty = max(cy_i - 5, th + 2)

    cv2.rectangle(img, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 2), (0, 0, 0), -1)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 0), thickness, cv2.LINE_AA)
