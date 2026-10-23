"""Parse a CS2 radar screenshot into world-space enemy/teammate positions.

The parser assumes the calibrated layout written by ``radar_calibrate.py``:

- circular radar centred on the local player;
- fixed orientation (``cl_radar_rotate 0``);
- world = self_world + R @ ((pixel - radar_center) / scale).

Enemy dots are red; teammate dots are any other high-saturation small blob.
Health and enemy alive count are not rendered on the radar, so they are
reported as unknown/defaults and flagged in ``warnings``.

Offline test:
    uv run --with pillow --with numpy --with scipy \
        python live/radar_parser.py --screenshot data/radar_capture/ingame_mid.jpg
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "live" / "models" / "radar_config_1280x960.json"
DEFAULT_AREA_GRID = PROJECT_ROOT / "live" / "models" / "area_grid.npz"

# The A/B bombsite labels are fixed world anchors. CS2 GSI does not expose the
# local player's position during normal play, so the parser recovers it from
# these two labels every frame.
BOMB_SITE_WORLD = {
    "A": np.array([1094.84, 2616.83]),
    "B": np.array([-1723.75, 2244.86]),
}


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = rgb.astype(float) / 255.0
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    maxc = array.max(axis=2)
    minc = array.min(axis=2)
    delta = maxc - minc
    saturation = np.divide(delta, maxc, out=np.zeros_like(delta), where=maxc > 1e-6)
    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    red_max = mask & (maxc == red)
    green_max = mask & (maxc == green) & ~red_max
    blue_max = mask & (maxc == blue) & ~red_max & ~green_max
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4
    hue = hue / 6.0
    return hue, saturation, maxc


def pixel_to_world(
    px: float, py: float, config: dict, self_world: tuple[float, float]
) -> tuple[float, float]:
    matrix = np.asarray(config["world_to_radar_matrix"], dtype=float)
    scale = float(config["scale"])
    if config.get("mode", "centered") == "static":
        origin = np.asarray(config["static_translation"], dtype=float)
        world = matrix.T @ ((np.array([px, py], dtype=float) - origin) / scale)
        return float(world[0]), float(world[1])

    center = np.asarray(config["radar_center"], dtype=float)
    offset = (np.array([px, py], dtype=float) - center) / scale
    world = matrix.T @ offset  # matrix is its own inverse (reflection)
    return float(self_world[0] + world[0]), float(self_world[1] + world[1])


def load_area_grid(path: Path | str = DEFAULT_AREA_GRID) -> dict | None:
    if not Path(path).exists():
        return None
    data = np.load(path)
    return {
        "grid": data["grid"],
        "x0": float(data["x0"]),
        "y0": float(data["y0"]),
        "cell": float(data["cell"]),
        "classes": [str(value) for value in data["classes"]],
    }


def classify_area(
    x: float,
    y: float,
    place_centroids: dict[str, dict],
    area_grid: dict | None = None,
) -> str:
    if area_grid is not None:
        grid = area_grid["grid"]
        ix = int((x - area_grid["x0"]) / area_grid["cell"])
        iy = int((y - area_grid["y0"]) / area_grid["cell"])
        if 0 <= ix < grid.shape[0] and 0 <= iy < grid.shape[1]:
            area = area_grid["classes"][int(grid[ix, iy])]
            return area if area in {"A", "B", "Mid", "CT"} else "other"
    if not place_centroids:
        return "other"
    best_name = min(
        place_centroids,
        key=lambda name: math.hypot(
            x - place_centroids[name]["x"], y - place_centroids[name]["y"]
        ),
    )
    area = place_centroids[best_name].get("area", "other")
    return area if area in {"A", "B", "Mid", "CT"} else "other"


def detect_site_labels(image: np.ndarray, config: dict) -> list[tuple[float, float]]:
    """Find the two A/B bomb-site labels on the radar.

    The labels are yellow/orange glyphs inside site boxes. Their pixel size
    scales with the radar radius, so the component filters are resolution
    aware. The two largest matching components are returned; the caller resolves
    which one is A vs B by checking which assignment puts the player inside the
    map.
    """
    roi = config["radar_roi"]
    cx, cy, radius = roi["cx"], roi["cy"], roi["r"]
    x0, y0 = int(roi.get("x0", 0)), int(roi.get("y0", 0))
    size = int(roi.get("size", math.ceil(cy + radius + 10)))
    crop = image[y0 : y0 + size, x0 : x0 + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        return []

    rgb = crop.astype(int)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    orange = (red > 130) & (green > 80) & (blue < 115) & (red > green + 15)
    yy, xx = np.mgrid[:size, :size]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 < (radius - 4) ** 2
    # The site labels rotate/pan with the map, so they are not always in the
    # lower half of the radar; do not filter by y.
    orange &= inside

    if radius >= 180:  # 2560x1440 layout
        area_range = (250, 1500)
        width_range = (12, 60)
        height_range = (12, 60)
    else:  # 1280x960 layout
        area_range = (40, 200)
        width_range = (6, 20)
        height_range = (9, 25)

    labels, _ = ndimage.label(orange)
    candidates: list[tuple[float, float, int]] = []
    for label_id, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(labels == label_id)
        area = len(xs)
        width = xs.max() - xs.min() + 1
        height = ys.max() - ys.min() + 1
        if not (area_range[0] <= area <= area_range[1]):
            continue
        if not (width_range[0] <= width <= width_range[1]):
            continue
        if not (height_range[0] <= height <= height_range[1]):
            continue
        candidates.append((float(xs.mean()), float(ys.mean()), area))
    candidates.sort(key=lambda item: item[2], reverse=True)
    return [(px + x0, py + y0) for px, py, _ in candidates[:2]]


def _grid_area(x: float, y: float, area_grid: dict | None) -> str | None:
    if area_grid is None:
        return None
    grid = area_grid["grid"]
    ix = int((x - area_grid["x0"]) / area_grid["cell"])
    iy = int((y - area_grid["y0"]) / area_grid["cell"])
    if not (0 <= ix < grid.shape[0] and 0 <= iy < grid.shape[1]):
        return None
    return area_grid["classes"][int(grid[ix, iy])]


def solve_from_labels(
    labels: list[tuple[float, float]],
    config: dict,
    area_grid: dict | None,
    prior_self: tuple[float, float] | None = None,
) -> dict | None:
    """Recover the player and map transform from the A/B label pixels.

    Tries both A/B assignments and keeps the one whose implied player position
    is inside the map. If both are valid, the candidate closest to the previous
    player position wins.
    """
    if len(labels) < 2:
        return None
    center = np.asarray(config["radar_center"], dtype=float)
    reflection = bool(config.get("fit", {}).get("reflection", True))
    a_world, b_world = BOMB_SITE_WORLD["A"], BOMB_SITE_WORLD["B"]
    world_delta = b_world - a_world

    candidates = []
    for swapped in (False, True):
        a_pixel, b_pixel = (
            (np.asarray(labels[1], float), np.asarray(labels[0], float))
            if swapped
            else (np.asarray(labels[0], float), np.asarray(labels[1], float))
        )
        pixel_delta = b_pixel - a_pixel
        scale = float(np.linalg.norm(pixel_delta) / np.linalg.norm(world_delta))
        if scale <= 0:
            continue
        # The radar zoom is fixed; a label pair implying a very different
        # scale is a false detection (e.g. an orange UI element).
        if abs(scale / float(config["scale"]) - 1.0) > 0.25:
            continue
        if reflection:
            theta = math.atan2(pixel_delta[1], pixel_delta[0]) + math.atan2(
                world_delta[1], world_delta[0]
            )
            matrix = np.array(
                [
                    [math.cos(theta), math.sin(theta)],
                    [math.sin(theta), -math.cos(theta)],
                ]
            )
        else:
            theta = math.atan2(pixel_delta[1], pixel_delta[0]) - math.atan2(
                world_delta[1], world_delta[0]
            )
            matrix = np.array(
                [
                    [math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)],
                ]
            )
        translation = a_pixel - scale * matrix @ a_world
        self_world = matrix.T @ ((center - translation) / scale)
        area = _grid_area(float(self_world[0]), float(self_world[1]), area_grid)
        if area is None:
            continue
        distance = (
            math.hypot(self_world[0] - prior_self[0], self_world[1] - prior_self[1])
            if prior_self is not None
            else 0.0
        )
        candidates.append(
            {
                "self_world": (float(self_world[0]), float(self_world[1])),
                "scale": scale,
                "matrix": matrix,
                "translation": translation,
                "area": area,
                "swapped": swapped,
                "distance": distance,
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["distance"], item["area"] == "other"))
    return candidates[0]


def pixel_to_world_transform(
    px: float, py: float, scale: float, matrix: np.ndarray, translation: np.ndarray
) -> tuple[float, float]:
    world = matrix.T @ ((np.array([px, py], dtype=float) - translation) / scale)
    return float(world[0]), float(world[1])


def _components(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    min_wh: int,
    max_wh: int,
) -> list[tuple[float, float, int]]:
    labels, _ = ndimage.label(mask)
    found: list[tuple[float, float, int]] = []
    for label_id, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(labels == label_id)
        size = len(xs)
        width = xs.max() - xs.min() + 1
        height = ys.max() - ys.min() + 1
        if not (min_area <= size <= max_area):
            continue
        if not (min_wh <= width <= max_wh and min_wh <= height <= max_wh):
            continue
        found.append((float(xs.mean()), float(ys.mean()), size))
    return found


def detect_dots(image: np.ndarray, config: dict) -> dict[str, list[dict]]:
    roi = config["radar_roi"]
    cx, cy, radius = roi["cx"], roi["cy"], roi["r"]
    x0, y0 = int(roi.get("x0", 0)), int(roi.get("y0", 0))
    size = int(roi.get("size", math.ceil(cy + radius + 10)))
    crop = image[y0 : y0 + size, x0 : x0 + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        raise ValueError("screenshot is smaller than the calibrated radar ROI")

    hue, saturation, value = rgb_to_hsv(crop)
    yy, xx = np.mgrid[:size, :size]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 < (radius - 3) ** 2

    enemy_cfg = config["colors"]["enemy"]
    red = (
        inside
        & (saturation >= enemy_cfg["s_min"])
        & (value >= enemy_cfg["v_min"])
        & ((hue <= enemy_cfg["h_max"]) | (hue >= 0.96))
    )

    teammate_cfg = config["colors"]["teammate"]
    teammate = (
        inside
        & (saturation >= teammate_cfg["s_min"])
        & (value >= teammate_cfg["v_min"])
        & ~red
    )

    enemy = red.copy()
    for x1, y1, x2, y2 in config.get("ui_exclude_boxes", []):
        enemy[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1] = False
        teammate[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1] = False

    # Dot size scales with the radar radius, so a 2560x1440 capture uses the
    # same thresholds as the 1280x960 reference calibration.
    dot_scale = radius / float(config.get("dot_reference_radius", 142.5))
    area_scale = dot_scale**2

    # Exclude the player's own marker at the radar centre.
    self_radius = float(config.get("self_exclude_radius", 8)) * dot_scale
    self_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= self_radius**2
    enemy &= ~self_mask
    teammate &= ~self_mask

    enemies = [
        {"px": px + x0, "py": py + y0, "size": area}
        for px, py, area in _components(
            enemy,
            max(8, int(30 * area_scale)),
            int(260 * area_scale),
            max(3, int(5 * dot_scale)),
            int(22 * dot_scale),
        )
    ]
    teammates = [
        {"px": px + x0, "py": py + y0, "size": area}
        for px, py, area in _components(
            teammate,
            max(5, int(teammate_cfg["area_min"] * area_scale)),
            int(teammate_cfg["area_max"] * area_scale),
            max(3, int(4 * dot_scale)),
            int(18 * dot_scale),
        )
    ]
    return {"enemies": enemies, "teammates": teammates}


def parse_frame(
    image: np.ndarray,
    config: dict,
    self_world: tuple[float, float] | None = None,
    self_area: str | None = None,
    prior_self: tuple[float, float] | None = None,
) -> dict:
    places = config.get("place_centroids", {})
    area_grid = load_area_grid()
    dots = detect_dots(image, config)
    warnings: list[str] = []

    located = None
    source = "gsi"
    if self_world is None or (self_world[0] == 0.0 and self_world[1] == 0.0):
        labels = detect_site_labels(image, config)
        if len(labels) >= 2:
            located = solve_from_labels(labels, config, area_grid, prior_self)
        if located is not None:
            self_world = located["self_world"]
            source = "radar_labels"
        else:
            warnings.append(
                "Could not locate the player from the radar A/B labels; "
                "rotation advice is withheld."
            )
            self_world = None

    if self_world is None:
        return {
            "enemies": [],
            "teammates": [],
            "self": None,
            "enemy_alive_count": 0,
            "self_spots_enemy": False,
            "warnings": warnings,
            "_transform": None,
        }

    def to_world(px: float, py: float) -> tuple[float, float]:
        if located is not None:
            return pixel_to_world_transform(
                px, py, located["scale"], located["matrix"], located["translation"]
            )
        return pixel_to_world(px, py, config, self_world)

    enemies = []
    for dot in dots["enemies"]:
        x, y = to_world(dot["px"], dot["py"])
        enemies.append(
            {
                "x": x,
                "y": y,
                "area": classify_area(x, y, places, area_grid),
                "visible": True,
                "health": 100,
            }
        )
    teammates = []
    for dot in dots["teammates"]:
        x, y = to_world(dot["px"], dot["py"])
        teammates.append(
            {
                "x": x,
                "y": y,
                "area": classify_area(x, y, places, area_grid),
                "health": 100,
            }
        )

    if teammates:
        warnings.append("Teammate health is not available from the radar; using 100.")
    enemy_alive_count = max(
        len(enemies), int(config.get("enemy_alive_default", len(enemies)))
    )
    warnings.append(
        "Enemy alive count is not available from the radar; assuming "
        f"{enemy_alive_count} alive."
    )
    return {
        "enemies": enemies,
        "teammates": teammates,
        "self": {
            "x": self_world[0],
            "y": self_world[1],
            "area": self_area
            or classify_area(self_world[0], self_world[1], places, area_grid),
            "source": source,
        },
        "enemy_alive_count": enemy_alive_count,
        "self_spots_enemy": False,
        "warnings": warnings,
        "_transform": {
            "scale": located["scale"],
            "matrix": located["matrix"].tolist(),
            "translation": located["translation"].tolist(),
        }
        if located is not None
        else None,
    }


def draw_overlay(image: np.ndarray, config: dict, parsed: dict) -> np.ndarray:
    output = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(output)
    roi = config["radar_roi"]
    draw.ellipse(
        [
            roi["cx"] - roi["r"],
            roi["cy"] - roi["r"],
            roi["cx"] + roi["r"],
            roi["cy"] + roi["r"],
        ],
        outline=(255, 0, 255),
        width=1,
    )
    if parsed.get("self") is None:
        return np.asarray(output)

    center = np.asarray(config["radar_center"], dtype=float)
    transform = parsed.get("_transform")
    if transform is not None:
        scale = float(transform["scale"])
        matrix = np.asarray(transform["matrix"], dtype=float)
        translation = np.asarray(transform["translation"], dtype=float)
    else:
        scale = float(config["scale"])
        matrix = np.asarray(config["world_to_radar_matrix"], dtype=float)
        translation = None
    self_world = (parsed["self"]["x"], parsed["self"]["y"])

    for kind, color in (("enemies", (255, 0, 0)), ("teammates", (0, 255, 0))):
        for item in parsed[kind]:
            world = np.array([item["x"], item["y"]])
            if translation is not None:
                pixel = translation + scale * matrix @ world
            else:
                offset = world - np.array(self_world)
                pixel = center + scale * matrix @ offset
            px, py = float(pixel[0]), float(pixel[1])
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], outline=color, width=2)
            draw.text((px + 10, py - 6), item["area"], fill=color)
    return np.asarray(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--self-x", type=float, default=None)
    parser.add_argument("--self-y", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    image = np.asarray(Image.open(args.screenshot).convert("RGB"))
    calibration_self = config["self_world_at_calibration"]
    self_world = (
        args.self_x if args.self_x is not None else calibration_self[0],
        args.self_y if args.self_y is not None else calibration_self[1],
    )
    parsed = parse_frame(image, config, self_world)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    overlay_path = args.output or args.screenshot.with_name(
        args.screenshot.stem + "_parsed.png"
    )
    Image.fromarray(draw_overlay(image, config, parsed)).save(overlay_path)
    print(f"wrote {overlay_path}")


if __name__ == "__main__":
    main()
