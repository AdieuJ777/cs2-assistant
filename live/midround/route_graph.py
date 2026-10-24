"""Dust2 CT route topology and exposure-aware shortest paths.

The graph is hand-maintained in ``live/models/dust2_ct_routes.json`` and is
kept deliberately small: it only needs enough nodes to describe macro
rotations between A / B / Mid / CT. Standard library only.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_ROUTES_PATH = Path(__file__).resolve().parents[1] / "models" / "dust2_ct_routes.json"

EXPOSURES = ("low", "medium", "high")
EXPOSURE_PENALTY = {
    "safe": {"low": 0.0, "medium": 4.0, "high": 20.0},
    "fast": {"low": 0.0, "medium": 0.5, "high": 1.5},
}
ENEMY_AREA_PENALTY = 15.0
ENEMY_NEAR_AREA_PENALTY = 2.0
TEAMMATE_AREA_BONUS = 1.5
ROUTE_TIME_MARGIN_S = 2.0


class RouteGraph:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = dict(data)
        self.map = str(data.get("map", "de_dust2"))
        self.side = str(data.get("side", "ct"))
        self.nodes: dict[str, dict[str, Any]] = {
            str(name): dict(attrs) for name, attrs in data["nodes"].items()
        }
        if not self.nodes:
            raise ValueError("route graph has no nodes")

        self.targets: dict[str, str] = {
            str(area): str(node) for area, node in (data.get("targets") or {}).items()
        }
        for area, node in self.targets.items():
            if node not in self.nodes:
                raise ValueError(f"target {area!r} points at unknown node {node!r}")

        self.adjacency: dict[str, list[dict[str, Any]]] = {
            node: [] for node in self.nodes
        }
        self._edges: list[dict[str, Any]] = []
        for raw in data.get("edges", []):
            edge = self._normalise_edge(raw)
            self._edges.append(edge)
            self.adjacency[edge["from"]].append(
                {"to": edge["to"], "eta_s": edge["eta_s"], "exposure": edge["exposure"]}
            )
            if edge["bidirectional"]:
                self.adjacency[edge["to"]].append(
                    {"to": edge["from"], "eta_s": edge["eta_s"], "exposure": edge["exposure"]}
                )

        self._area_nodes: dict[str, list[str]] = {}
        for name, attrs in self.nodes.items():
            area = str(attrs.get("area", "other"))
            self._area_nodes.setdefault(area, []).append(name)

        # Macro-area adjacency, used to make a visible enemy raise the risk of
        # the neighbouring macro areas as well.
        self._area_adjacency: dict[str, set[str]] = {}
        for edge in self._edges:
            area_from = self.node_area(edge["from"])
            area_to = self.node_area(edge["to"])
            if area_from == area_to:
                continue
            self._area_adjacency.setdefault(area_from, set()).add(area_to)
            self._area_adjacency.setdefault(area_to, set()).add(area_from)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Path | str | None = None) -> "RouteGraph":
        source = Path(path) if path is not None else DEFAULT_ROUTES_PATH
        return cls(json.loads(source.read_text(encoding="utf-8")))

    @staticmethod
    def _normalise_edge(raw: Mapping[str, Any]) -> dict[str, Any]:
        for key in ("from", "to"):
            if key not in raw:
                raise ValueError(f"edge is missing {key!r}: {raw!r}")
        eta = float(raw.get("eta_s", 0.0))
        if eta <= 0.0:
            raise ValueError(f"edge eta_s must be positive: {raw!r}")
        exposure = str(raw.get("exposure", "low"))
        if exposure not in EXPOSURES:
            raise ValueError(f"unknown exposure {exposure!r}: {raw!r}")
        return {
            "from": str(raw["from"]),
            "to": str(raw["to"]),
            "eta_s": eta,
            "exposure": exposure,
            "bidirectional": bool(raw.get("bidirectional", True)),
        }

    # --------------------------------------------------------------- helpers
    def __contains__(self, node: object) -> bool:
        return node in self.nodes

    def node_area(self, node: str) -> str:
        return str(self.nodes[node].get("area", "other"))

    def callout(self, node: str) -> str:
        return str(self.nodes[node].get("callout", node))

    def nodes_in_area(self, area: str) -> list[str]:
        return list(self._area_nodes.get(area, []))

    def resolve_start(
        self, area: str | None, x: float | None = None, y: float | None = None
    ) -> str | None:
        """Pick a concrete start node inside a macro area.

        Uses the nearest node when world coordinates are available, otherwise
        the area anchor (``targets``) and finally the first node in the area.
        """
        if area in self.nodes:
            return area
        if not area:
            # No macro area yet: fall back to the nearest node in the whole
            # graph when a world position is available.
            if x is not None and y is not None:
                return self.nearest_node(self.nodes, x, y)
            return None
        candidates = self.nodes_in_area(area)
        if not candidates:
            if x is not None and y is not None:
                return self.nearest_node(self.nodes, x, y)
            return None
        if x is not None and y is not None:
            return self.nearest_node(candidates, x, y)
        anchor = self.targets.get(area)
        return anchor if anchor in candidates else candidates[0]

    def nearest_node(self, candidates: Iterable[str], x: float, y: float) -> str:
        def distance(node: str) -> float:
            attrs = self.nodes[node]
            return math.hypot(
                x - float(attrs.get("x", 0.0)), y - float(attrs.get("y", 0.0))
            )

        return min(candidates, key=distance)

    def is_path(self, route: Iterable[str]) -> bool:
        path = list(route)
        if not path:
            return False
        if any(node not in self.nodes for node in path):
            return False
        for current, following in zip(path, path[1:]):
            if not any(edge["to"] == following for edge in self.adjacency[current]):
                return False
        return True

    def path_eta(self, route: Iterable[str]) -> float:
        path = list(route)
        total = 0.0
        for current, following in zip(path, path[1:]):
            matches = [
                edge for edge in self.adjacency[current] if edge["to"] == following
            ]
            if not matches:
                raise ValueError(f"no edge {current!r} -> {following!r}")
            total += float(matches[0]["eta_s"])
        return total

    # ---------------------------------------------------------------- search
    def shortest_path(
        self,
        start: str,
        target: str,
        *,
        style: str = "safe",
        enemy_areas: Iterable[str] = (),
        teammate_areas: Iterable[str] = (),
        time_left: float | None = None,
    ) -> dict[str, Any] | None:
        if style not in EXPOSURE_PENALTY:
            raise ValueError(f"unknown route style {style!r}")
        if start not in self.nodes:
            raise ValueError(f"unknown start node {start!r}")

        anchors = self._target_anchors(target)
        if not anchors:
            return None

        enemy_set = {str(area) for area in enemy_areas}
        enemy_adjacent: set[str] = set()
        for area in enemy_set:
            enemy_adjacent.update(self._area_adjacency.get(area, set()))
        enemy_adjacent -= enemy_set
        teammate_set = {str(area) for area in teammate_areas}
        exposure_penalty = EXPOSURE_PENALTY[style]

        best: dict[str, Any] | None = None
        for anchor in anchors:
            result = self._dijkstra(
                start,
                anchor,
                style=style,
                exposure_penalty=exposure_penalty,
                enemy_set=enemy_set,
                enemy_adjacent=enemy_adjacent,
                teammate_set=teammate_set,
            )
            if result is None:
                continue
            if best is None or (result["cost"], result["eta_s"]) < (
                best["cost"],
                best["eta_s"],
            ):
                best = result

        if best is None:
            return None
        if time_left is not None and best["eta_s"] + ROUTE_TIME_MARGIN_S > float(time_left):
            return None
        return best

    def find_routes(
        self,
        start: str,
        target: str,
        *,
        enemy_areas: Iterable[str] = (),
        teammate_areas: Iterable[str] = (),
        time_left: float | None = None,
    ) -> dict[str, dict[str, Any] | None]:
        """Return the ``safe`` and ``fast`` variants for one destination.

        Identical paths collapse to ``{"safe": route, "fast": None}``.
        Routes that cannot be completed before the round ends are dropped.
        """
        common = {
            "enemy_areas": tuple(enemy_areas),
            "teammate_areas": tuple(teammate_areas),
            "time_left": time_left,
        }
        safe = self.shortest_path(start, target, style="safe", **common)
        fast = self.shortest_path(start, target, style="fast", **common)
        if safe is not None and fast is not None and safe["route"] == fast["route"]:
            fast = None
        return {"safe": safe, "fast": fast}

    def _target_anchors(self, target: str) -> list[str]:
        if target in self.nodes:
            return [target]
        anchor = self.targets.get(str(target))
        return [anchor] if anchor in self.nodes else []

    def _dijkstra(
        self,
        start: str,
        target: str,
        *,
        style: str,
        exposure_penalty: Mapping[str, float],
        enemy_set: set[str],
        enemy_adjacent: set[str],
        teammate_set: set[str],
    ) -> dict[str, Any] | None:
        def node_penalty(node: str) -> float:
            area = self.node_area(node)
            if area in enemy_set:
                return ENEMY_AREA_PENALTY
            if area in enemy_adjacent:
                return ENEMY_NEAR_AREA_PENALTY
            return 0.0

        def node_bonus(node: str) -> float:
            return -TEAMMATE_AREA_BONUS if self.node_area(node) in teammate_set else 0.0

        distances: dict[str, float] = {start: 0.0}
        previous: dict[str, str] = {}
        queue: list[tuple[float, str]] = [(0.0, start)]
        visited: set[str] = set()

        while queue:
            cost, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            if node == target:
                break
            for edge in self.adjacency[node]:
                neighbour = edge["to"]
                if neighbour in visited:
                    continue
                weight = (
                    float(edge["eta_s"])
                    + exposure_penalty[str(edge["exposure"])]
                    + node_penalty(neighbour)
                    + node_bonus(neighbour)
                )
                weight = max(0.05, weight)
                candidate = cost + weight
                if candidate < distances.get(neighbour, math.inf):
                    distances[neighbour] = candidate
                    previous[neighbour] = node
                    heapq.heappush(queue, (candidate, neighbour))

        if target not in distances:
            return None

        path = [target]
        while path[-1] != start:
            parent = previous.get(path[-1])
            if parent is None:
                return None
            path.append(parent)
        path.reverse()

        eta = self.path_eta(path)
        exposures = [
            str(edge["exposure"])
            for current, following in zip(path, path[1:])
            for edge in self.adjacency[current]
            if edge["to"] == following
        ]
        worst = max(exposures, key=EXPOSURES.index) if exposures else "low"
        return {
            "route": path,
            "target": target,
            "eta_s": round(eta, 1),
            "exposure": worst,
            "cost": round(distances[target], 2),
            "style": style,
        }
