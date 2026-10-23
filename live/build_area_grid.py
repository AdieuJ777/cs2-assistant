"""Build a world-coordinate -> macro-area lookup from parsed demo ticks.

Nearest-place-centroid misclassifies boundary positions (e.g. top-mid vs
catwalk). A majority-vote grid over millions of parsed positions is more
faithful to the Awpy place labels used to train the rotation policy.

Output: ``live/models/area_grid.npz`` with ``grid`` (uint8 codes), ``x0``,
``y0``, ``cell`` and ``classes``.

Run:
    uv run --with numpy --with pandas --with pyarrow \
        python live/build_area_grid.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSED_ROOT = PROJECT_ROOT.parent / "TAGNet-CS2-main" / "data" / "parsed"
DEFAULT_OUTPUT = PROJECT_ROOT / "live" / "models" / "area_grid.npz"

_spec = importlib.util.spec_from_file_location(
    "enemy_build_dataset",
    PROJECT_ROOT / "benchmarks" / "enemy_visibility" / "build_dataset.py",
)
build_dataset = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(build_dataset)

CLASSES = ["A", "B", "Mid", "CT", "other"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cell", type=int, default=32)
    args = parser.parse_args()

    frames = []
    for folder in sorted(PARSED_ROOT.iterdir()):
        ticks = folder / "ticks.parquet"
        if ticks.exists():
            data = pd.read_parquet(ticks, columns=["X", "Y", "place"])
            frames.append(data.dropna(subset=["X", "Y"]))
    if not frames:
        raise SystemExit("no parsed ticks found")
    positions = pd.concat(frames, ignore_index=True)
    positions["area"] = positions["place"].map(build_dataset.map_area)

    x0, y0 = -2600.0, -1300.0
    x1, y1 = 2000.0, 3500.0
    cell = float(args.cell)
    nx = int(np.ceil((x1 - x0) / cell)) + 1
    ny = int(np.ceil((y1 - y0) / cell)) + 1
    ix = np.clip(((positions["X"] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((positions["Y"] - y0) / cell).astype(int), 0, ny - 1)

    counts = np.zeros((nx, ny, len(CLASSES)), dtype=np.int32)
    for code, name in enumerate(CLASSES):
        mask = positions["area"].to_numpy() == name
        if mask.any():
            counts[:, :, code] = np.bincount(
                ix[mask] * ny + iy[mask], minlength=nx * ny
            ).reshape(nx, ny)

    grid = counts.argmax(axis=2).astype(np.uint8)
    grid[counts.sum(axis=2) == 0] = 255
    # Fill empty cells from the nearest observed cell.
    known = grid != 255
    if known.any():
        _, indices = ndimage.distance_transform_edt(
            ~known, return_indices=True
        )
        grid[~known] = grid[indices[0][~known], indices[1][~known]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        grid=grid,
        x0=np.float64(x0),
        y0=np.float64(y0),
        cell=np.float64(cell),
        classes=np.array(CLASSES),
    )
    # A JSON copy lets the stdlib live server resolve player areas without
    # numpy/scipy.
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "x0": x0,
                "y0": y0,
                "cell": cell,
                "nx": int(nx),
                "ny": int(ny),
                "classes": CLASSES,
                "grid": grid.ravel().tolist(),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(
        f"{len(positions):,} positions, {nx}x{ny} cells, "
        f"unknown={int((~known).sum())} -> {args.output}"
    )
    print(f"wrote {json_path}")
    for code, name in enumerate(CLASSES):
        print(f"  {name}: {int((grid == code).sum())} cells")


if __name__ == "__main__":
    main()
