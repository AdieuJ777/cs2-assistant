"""Tests for the hand-maintained Dust2 CT route graph."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.midround.route_graph import RouteGraph  # noqa: E402


class RouteGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = RouteGraph.load()

    def test_loads_nodes_and_targets(self) -> None:
        self.assertIn("Mid", self.graph.nodes)
        self.assertIn("B_Site", self.graph.nodes)
        for area in ("A", "B", "Mid", "CT"):
            self.assertIn(area, self.graph.targets)
            self.assertIn(self.graph.targets[area], self.graph.nodes)

    def test_nearest_start_node(self) -> None:
        start = self.graph.resolve_start("A", 1430.0, 1550.0)
        self.assertIn(start, self.graph.nodes_in_area("A"))
        self.assertEqual(start, self.graph.resolve_start("A", 1430.0, 1550.0))

    def test_start_falls_back_to_coordinates_without_area(self) -> None:
        start = self.graph.resolve_start(None, -346.0, 1345.0)
        self.assertEqual(start, "Mid")

    def test_safe_and_fast_routes_differ(self) -> None:
        routes = self.graph.find_routes("Mid", "B")
        safe = routes["safe"]
        fast = routes["fast"]
        self.assertIsNotNone(safe)
        self.assertIsNotNone(fast)
        self.assertEqual(safe["route"][0], "Mid")
        self.assertEqual(safe["route"][-1], "B_Site")
        self.assertEqual(fast["route"][-1], "B_Site")
        self.assertNotEqual(safe["route"], fast["route"])
        self.assertLessEqual(fast["eta_s"], safe["eta_s"])

    def test_safe_route_avoids_high_exposure(self) -> None:
        safe = self.graph.find_routes("Mid", "B")["safe"]
        self.assertIsNotNone(safe)
        self.assertNotEqual(safe["exposure"], "high")

    def test_time_constraint_drops_route(self) -> None:
        routes = self.graph.find_routes("Mid", "B", time_left=5.0)
        self.assertIsNone(routes["safe"])
        self.assertIsNone(routes["fast"])

    def test_enemy_area_is_avoided(self) -> None:
        data = {
            "nodes": {
                "S": {"area": "CT"},
                "X": {"area": "A"},
                "Y": {"area": "CT"},
                "T": {"area": "B"},
            },
            "targets": {"B": "T"},
            "edges": [
                {"from": "S", "to": "X", "eta_s": 3.0, "exposure": "low"},
                {"from": "X", "to": "T", "eta_s": 3.0, "exposure": "low"},
                {"from": "S", "to": "Y", "eta_s": 5.0, "exposure": "low"},
                {"from": "Y", "to": "T", "eta_s": 5.0, "exposure": "low"},
            ],
        }
        graph = RouteGraph(data)
        routes = graph.find_routes("S", "T", enemy_areas={"A"})
        self.assertIsNotNone(routes["safe"])
        self.assertNotIn("X", routes["safe"]["route"])
        self.assertEqual(routes["safe"]["route"], ["S", "Y", "T"])

    def test_unknown_target_returns_none(self) -> None:
        self.assertIsNone(self.graph.shortest_path("Mid", "Nowhere"))

    def test_path_helpers(self) -> None:
        route = self.graph.shortest_path("Mid", "B_Site", style="fast")
        self.assertTrue(self.graph.is_path(route["route"]))
        self.assertAlmostEqual(
            self.graph.path_eta(route["route"]), route["eta_s"], places=1
        )
        self.assertFalse(self.graph.is_path(["Mid", "B_Site"]))
        self.assertFalse(self.graph.is_path([]))

    def test_invalid_edge_is_rejected(self) -> None:
        data = {
            "nodes": {"A": {"area": "A"}, "B": {"area": "B"}},
            "edges": [{"from": "A", "to": "B", "eta_s": 0.0}],
        }
        with self.assertRaises(ValueError):
            RouteGraph(data)

    def test_unknown_target_area_rejected(self) -> None:
        data = {
            "nodes": {"A": {"area": "A"}},
            "targets": {"X": "Missing"},
            "edges": [],
        }
        with self.assertRaises(ValueError):
            RouteGraph(data)


if __name__ == "__main__":
    unittest.main()
