"""Calibrate a CS2 radar screenshot to Dust2 world coordinates.

The v1 radar default is ``cl_radar_always_centered 1`` and
``cl_radar_rotate 0``:

- the player marker is the centre of the circular radar;
- the map is a fixed, mirrored view of the Awpy world coordinates;
- pixel -> world is ``self_world + R @ ((pixel - center) / scale)``.

Calibration uses the two bombsite labels (A/B) plus the Dust2 walkable mask:
their pixel centres give scale and rotation, and the two mirror solutions are
scored against the map silhouette.

Run:
    uv run --with pillow --with numpy --with scipy \
        python live/radar_calibrate.py \
        --screenshot data/radar_capture/ingame_mid.jpg \
        --output live/models/radar_config_1280x960.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAGNET_ROOT = PROJECT_ROOT.parent / "TAGNet-CS2-main"
DEFAULT_MASK = TAGNET_ROOT / "data" / "map" / "dust2_walkable_mask.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "live" / "models" / "radar_config_1280x960.json"
DEFAULT_PLACES = PROJECT_ROOT / "live" / "models" / "place_centroids.json"

# Median positions from the parsed demo ticks.
BOMB_SITE_WORLD = {
    "A": np.array([1094.84, 2616.83]),
    "B": np.array([-1723.75, 2244.86]),
}


def find_radar_disk(image: np.ndarray) -> tuple[float, float, float]:
    # The radar sits in the top-left corner; a larger region is needed at
    # 2560x1440 because the radar is physically larger there.
    wanted = 700 if image.shape[1] >= 1600 else 500
    search = min(wanted, image.shape[0], image.shape[1])
    rgb = image[:search, :search].astype(int)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ring = (
        (red > 180)
        & (green > 110)
        & (blue > 130)
        & (red - green > 25)
        & (blue - green > 5)
    )
    ys, xs = np.nonzero(ring)
    if len(xs) < 50:
        raise SystemExit("could not find the radar ring")
    # At 2560x1440 the ring is dense enough for outlier rejection. At
    # 1280x960 the ring has far fewer pixels, so a plain fit is more stable.
    keep = np.ones(len(xs), dtype=bool)
    if len(xs) >= 1000:
        for _ in range(5):
            x, y = xs[keep], ys[keep]
            a = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
            b = x**2 + y**2
            solution, *_ = np.linalg.lstsq(a, b, rcond=None)
            cx, cy = solution[0], solution[1]
            radius = math.sqrt(solution[2] + cx**2 + cy**2)
            keep_new = np.abs(np.hypot(xs - cx, ys - cy) - radius) < 6
            if keep_new.sum() < 50:
                break
            keep = keep_new
    else:
        x, y = xs, ys
        a = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
        b = x**2 + y**2
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy = solution[0], solution[1]
        radius = math.sqrt(solution[2] + cx**2 + cy**2)
    return float(cx), float(cy), float(radius)


def find_site_labels(
    image: np.ndarray, cx: float, cy: float, radius: float
) -> dict[str, np.ndarray]:
    rgb = image.astype(int)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    orange = (red > 130) & (green > 80) & (blue < 100) & (red > green + 15)
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 < (radius - 4) ** 2
    orange &= inside & (yy > cy - radius * 0.2)

    labels, _ = ndimage.label(orange)
    candidates = []
    for label_id, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(labels == label_id)
        size = len(xs)
        width = xs.max() - xs.min() + 1
        height = ys.max() - ys.min() + 1
        # The letter itself is a compact 6-12 x 9-16 glyph; the site box edges
        # are taller/wider and would shift the centroid.
        if 40 <= size <= 110 and 6 <= width <= 12 and 9 <= height <= 17:
            candidates.append(
                (float(xs.mean()), float(ys.mean()), size, width, height)
            )
    left = [c for c in candidates if c[0] < cx - 20]
    right = [c for c in candidates if c[0] > cx + 20]
    if not left or not right:
        raise SystemExit("could not find both A and B radar labels")
    # The site box edge is another orange glyph; the label itself is the one
    # nearest the radar centre horizontally.
    left.sort(key=lambda c: abs(c[0] - cx))
    right.sort(key=lambda c: abs(c[0] - cx))
    return {"A": np.array(left[0][:2]), "B": np.array(right[0][:2])}


def similarity(
    a_world: np.ndarray,
    b_world: np.ndarray,
    a_pixel: np.ndarray,
    b_pixel: np.ndarray,
    reflect: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    world_delta = b_world - a_world
    pixel_delta = b_pixel - a_pixel
    scale = np.linalg.norm(pixel_delta) / np.linalg.norm(world_delta)
    if reflect:
        theta = math.atan2(pixel_delta[1], pixel_delta[0]) + math.atan2(
            world_delta[1], world_delta[0]
        )
        matrix = np.array(
            [[math.cos(theta), math.sin(theta)], [math.sin(theta), -math.cos(theta)]]
        )
    else:
        theta = math.atan2(pixel_delta[1], pixel_delta[0]) - math.atan2(
            world_delta[1], world_delta[0]
        )
        matrix = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
    translation = a_pixel - scale * matrix @ a_world
    return float(scale), matrix, translation


def silhouette_iou(
    image: np.ndarray, cx: float, cy: float, radius: float, scale: float, matrix: np.ndarray, translation: np.ndarray
) -> float:
    data = np.load(DEFAULT_MASK)
    mask = data["mask"].astype(bool)
    x_edges = data["x_edges"].astype(float)
    y_edges = data["y_edges"].astype(float)
    cell = float(x_edges[1] - x_edges[0])

    crop = image.astype(float)
    gray = crop.mean(axis=2)
    saturation = crop.max(axis=2) - crop.min(axis=2)
    yy, xx = np.mgrid[: crop.shape[0], : crop.shape[1]]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 < (radius - 6) ** 2
    radar_walk = inside & (gray > 135) & (saturation < 40)
    ys, xs = np.nonzero(inside)
    px, py = xs.astype(float), ys.astype(float)

    # pixel = scale * R @ world + t  ->  world = R^T @ ((pixel - t) / scale)
    inv = matrix.T
    dx = (px - translation[0]) / scale
    dy = (py - translation[1]) / scale
    world_x = inv[0, 0] * dx + inv[0, 1] * dy
    world_y = inv[1, 0] * dx + inv[1, 1] * dy
    ix = ((world_x - x_edges[0]) / cell).astype(int)
    iy = ((world_y - y_edges[0]) / cell).astype(int)
    valid = (ix >= 0) & (ix < mask.shape[0]) & (iy >= 0) & (iy < mask.shape[1])
    predicted = np.zeros(len(px), dtype=bool)
    predicted[valid] = mask[ix[valid], iy[valid]]
    intersection = np.sum(predicted & radar_walk[ys, xs])
    union = np.sum(predicted | radar_walk[ys, xs])
    return float(intersection / max(union, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--places", type=Path, default=DEFAULT_PLACES)
    parser.add_argument(
        "--mode",
        choices=["centered", "static"],
        default="centered",
        help="centered = cl_radar_always_centered 1; static = 0.",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Mark the radar as rotating; the parser does not handle this yet.",
    )
    parser.add_argument(
        "--a-px",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Manual A label pixel centre (skips automatic label detection).",
    )
    parser.add_argument(
        "--b-px",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Manual B label pixel centre (skips automatic label detection).",
    )
    args = parser.parse_args()

    image = np.asarray(Image.open(args.screenshot).convert("RGB"))
    cx, cy, radius = find_radar_disk(image)
    if args.a_px is not None and args.b_px is not None:
        labels = {
            "A": np.array(args.a_px, dtype=float),
            "B": np.array(args.b_px, dtype=float),
        }
    else:
        labels = find_site_labels(image, cx, cy, radius)

    candidates = []
    for reflect in (False, True):
        scale, matrix, translation = similarity(
            BOMB_SITE_WORLD["A"],
            BOMB_SITE_WORLD["B"],
            labels["A"],
            labels["B"],
            reflect,
        )
        iou = silhouette_iou(image, cx, cy, radius, scale, matrix, translation)
        self_world = matrix.T @ ((np.array([cx, cy]) - translation) / scale)
        # Sanity: the player at the radar centre must land inside the map.
        plausible = -2000 <= self_world[0] <= 2000 and -1000 <= self_world[1] <= 3000
        candidates.append(
            {
                "reflect": reflect,
                "scale": scale,
                "matrix": matrix,
                "translation": translation,
                "iou": iou,
                "self_world": self_world,
                "plausible": plausible,
            }
        )
    candidates = [c for c in candidates if c["plausible"]] or candidates
    best = max(candidates, key=lambda c: c["iou"])

    places = json.loads(args.places.read_text(encoding="utf-8")) if args.places.exists() else {}
    config = {
        "version": 2,
        "mode": args.mode,
        "rotate": args.rotate,
        "source_screenshot": str(args.screenshot),
        "resolution": [int(image.shape[1]), int(image.shape[0])],
        "radar_roi": {
            "cx": cx,
            "cy": cy,
            "r": radius,
            "x0": 0,
            "y0": 0,
            "size": int(math.ceil(cy + radius + 10)),
        },
        "scale": best["scale"],
        "world_to_radar_matrix": best["matrix"].tolist(),
        "radar_center": [cx, cy],
        "static_translation": best["translation"].tolist(),
        "self_world_at_calibration": best["self_world"].tolist(),
        "place_centroids": places,
        "labels_px": {key: value.tolist() for key, value in labels.items()},
        "fit": {
            "iou": best["iou"],
            "reflection": best["reflect"],
            "note": "chosen by silhouette IoU + map plausibility",
        },
        "colors": {
            "enemy": {"h_min": 0.0, "h_max": 0.04, "s_min": 0.55, "v_min": 0.35},
            "teammate": {"s_min": 0.45, "v_min": 0.35, "area_min": 15, "area_max": 180},
        },
        "ui_exclude_boxes": [
            [
                float(labels[key][0] - 18 * radius / 142.5),
                float(labels[key][1] - 18 * radius / 142.5),
                float(labels[key][0] + 18 * radius / 142.5),
                float(labels[key][1] + 18 * radius / 142.5),
            ]
            for key in ("A", "B")
        ],
        "self_exclude_radius": 8,
        "enemy_alive_default": 5,
        "notes": (
            "Centered radar (cl_radar_always_centered 1). rotate=false; if "
            "cl_radar_rotate is 1, recalibrate with the player facing a known "
            "direction or set cl_radar_rotate 0."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"center=({cx:.1f},{cy:.1f}) r={radius:.1f} scale={best['scale']:.5f} "
        f"reflect={best['reflect']} iou={best['iou']:.3f} "
        f"self_world={np.round(best['self_world'], 0)}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
