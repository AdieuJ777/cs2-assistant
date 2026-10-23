"""World-coordinate -> macro-area lookup for the stdlib live server.

Uses the majority-vote grid produced by ``build_area_grid.py`` (JSON copy), or
the place centroids as a fallback. Kept dependency-free so the server can run
without numpy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

AREAS = {"A", "B", "Mid", "CT"}


class AreaLookup:
    def __init__(
        self,
        *,
        grid: list[int] | None = None,
        x0: float = 0.0,
        y0: float = 0.0,
        cell: float = 32.0,
        nx: int = 0,
        ny: int = 0,
        classes: list[str] | None = None,
        places: dict[str, dict] | None = None,
    ) -> None:
        self.grid = grid
        self.x0 = x0
        self.y0 = y0
        self.cell = cell
        self.nx = nx
        self.ny = ny
        self.classes = classes or []
        self.places = places or {}

    @classmethod
    def load(cls, grid_path: Path | str, places_path: Path | str | None = None) -> "AreaLookup":
        grid_file = Path(grid_path)
        if grid_file.exists():
            data = json.loads(grid_file.read_text(encoding="utf-8"))
            return cls(
                grid=data["grid"],
                x0=float(data["x0"]),
                y0=float(data["y0"]),
                cell=float(data["cell"]),
                nx=int(data["nx"]),
                ny=int(data["ny"]),
                classes=list(data["classes"]),
            )
        if places_path is not None and Path(places_path).exists():
            return cls(places=json.loads(Path(places_path).read_text(encoding="utf-8")))
        return cls()

    def area(self, x: float, y: float) -> str | None:
        if self.grid is not None:
            ix = int((x - self.x0) / self.cell)
            iy = int((y - self.y0) / self.cell)
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                code = self.grid[ix * self.ny + iy]
                name = self.classes[code] if code < len(self.classes) else "other"
                return name if name in AREAS else "other"
            return None
        if self.places:
            name = min(
                self.places,
                key=lambda key: math.hypot(
                    x - self.places[key]["x"], y - self.places[key]["y"]
                ),
            )
            area = self.places[name].get("area", "other")
            return area if area in AREAS else "other"
        return None
