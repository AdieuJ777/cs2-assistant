"""Tests for mid-round candidate generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.midround.candidate_builder import (  # noqa: E402
    MAX_CALL_CHARS,
    build_candidates,
    validate_candidate,
)
from live.midround.route_graph import RouteGraph  # noqa: E402
from live.midround.state_card import build_state_card  # noqa: E402


def gsi_payload(
    *,
    phase: str = "live",
    health: int = 100,
    seconds_left: float | None = 38.0,
    bomb: str = "carried",
    bomb_site: str | None = None,
    position: dict | None = None,
) -> dict:
    payload = {
        "map": {
            "name": "de_dust2",
            "round": 5,
            "team_ct": {"score": 5},
            "team_t": {"score": 3},
        },
        "round": {"phase": phase, "bomb": bomb},
        "player": {
            "team": "CT",
            "state": {
                "health": health,
                "armor": 100,
                "equipment_value": 4200,
                "defusekit": True,
            },
            "position": position if position is not None else {"x": -346.0, "y": 1345.0},
            "forward": {"x": 1.0, "y": 0.0},
        },
        "phase_countdowns": (
            {"phase_ends_in": str(seconds_left)} if seconds_left is not None else {}
        ),
    }
    if bomb_site is not None:
        payload["round"]["bomb_site"] = bomb_site
    return payload


def radar_payload(*, teammates: list | None = None, enemies: list | None = None) -> dict:
    payload = {
        "teammates": [
            {"x": -1724.0, "y": 2245.0, "health": 80, "area": "B"},
            {"x": 1094.0, "y": 2617.0, "health": 45, "area": "A"},
        ]
        if teammates is None
        else teammates,
        "enemies": [{"x": 1094.0, "y": 2617.0, "area": "A", "visible": True}]
        if enemies is None
        else enemies,
        "enemy_alive_count": 5,
        "self_spots_enemy": True,
    }
    return payload


def rotation(action: str = "ROTATE_B", warnings: list | None = None) -> dict:
    return {
        "action": action,
        "destination": action,
        "stage1_proba": 0.64,
        "destination_proba": 0.72,
        "threshold": 0.6,
        "warnings": warnings or [],
    }


def make_card(**overrides) -> dict:
    gsi = overrides.pop("gsi", gsi_payload())
    radar = overrides.pop("radar", radar_payload())
    rot = overrides.pop("rotation", rotation())
    return build_state_card(gsi, radar, rot, now=1760000000.0, radar_age_s=0.1)


class CandidateBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = RouteGraph.load()

    def assert_valid_list(self, card: dict, candidates: list[dict]) -> None:
        self.assertGreaterEqual(len(candidates), 1)
        self.assertLessEqual(len(candidates), 4)
        ids = [candidate["id"] for candidate in candidates]
        self.assertEqual(len(ids), len(set(ids)))
        for candidate in candidates:
            ok, reason = validate_candidate(candidate, card, self.graph)
            self.assertTrue(ok, f"{candidate['id']}: {reason}")
            self.assertLessEqual(len(candidate["call_hint"]), MAX_CALL_CHARS)

    def test_rotate_b_primary_is_safe_route(self) -> None:
        card = make_card()
        candidates = build_candidates(card, self.graph)
        self.assert_valid_list(card, candidates)
        self.assertEqual(candidates[0]["action"], "ROTATE_B")
        self.assertEqual(candidates[0]["target_area"], "B")
        self.assertEqual(candidates[0]["route_style"], "safe")
        self.assertEqual(candidates[0]["route"][0], "Mid")

    def test_rotate_b_offers_fast_when_it_differs(self) -> None:
        # No visible enemy: the exposure-only fast route differs from the
        # safe route, so both variants are offered.
        card = make_card(
            radar=radar_payload(enemies=[]), rotation=rotation("ROTATE_B")
        )
        candidates = build_candidates(card, self.graph)
        self.assert_valid_list(card, candidates)
        styles = {candidate.get("route_style") for candidate in candidates}
        self.assertIn("safe", styles)
        self.assertIn("fast", styles)

    def test_hold_scenario(self) -> None:
        card = make_card(rotation=rotation("HOLD"))
        candidates = build_candidates(card, self.graph)
        self.assert_valid_list(card, candidates)
        self.assertEqual(candidates[0]["action"], "HOLD")
        self.assertEqual(candidates[0]["target_area"], "Mid")
        self.assertEqual(candidates[0]["eta_s"], 0.0)

    def test_last_player_gets_fallback(self) -> None:
        card = make_card(radar=radar_payload(teammates=[]))
        candidates = build_candidates(card, self.graph)
        self.assert_valid_list(card, candidates)
        actions = [candidate["action"] for candidate in candidates]
        self.assertIn("FALL_BACK", actions)
        fallback = next(c for c in candidates if c["action"] == "FALL_BACK")
        self.assertEqual(fallback["target_area"], "CT")

    def test_low_health_gets_fallback(self) -> None:
        card = make_card(gsi=gsi_payload(health=25))
        candidates = build_candidates(card, self.graph)
        self.assert_valid_list(card, candidates)
        self.assertIn("FALL_BACK", [candidate["action"] for candidate in candidates])

    def test_insufficient_time_drops_rotations(self) -> None:
        card = make_card(gsi=gsi_payload(seconds_left=5.0))
        candidates = build_candidates(card, self.graph)
        self.assert_valid_list(card, candidates)
        actions = [candidate["action"] for candidate in candidates]
        self.assertNotIn("ROTATE_B", actions)
        self.assertIn("WAIT", actions)
        for candidate in candidates:
            self.assertLessEqual(candidate["eta_s"] + 2.0, 5.0 + 1e-9)

    def test_no_enemy_information_waits(self) -> None:
        card = make_card(
            radar=radar_payload(enemies=[]),
            rotation=rotation("NO_RADAR", warnings=["No visible enemy"]),
        )
        candidates = build_candidates(card, self.graph)
        self.assert_valid_list(card, candidates)
        self.assertEqual(candidates[0]["action"], "WAIT")
        self.assertIn("HOLD", [candidate["action"] for candidate in candidates])

    def test_bomb_planted_facts(self) -> None:
        card = make_card(
            gsi=gsi_payload(bomb="planted", bomb_site="A", seconds_left=18.0),
            rotation=rotation("HOLD"),
        )
        candidates = build_candidates(card, self.graph)
        self.assert_valid_list(card, candidates)
        facts = {fact for candidate in candidates for fact in candidate["grounded_facts"]}
        self.assertIn("炸弹已下在 A", facts)

    def test_grounded_facts_come_from_card(self) -> None:
        card = make_card()
        candidates = build_candidates(card, self.graph)
        for candidate in candidates:
            for fact in candidate["grounded_facts"]:
                self.assertNotIn("一定", fact)
                self.assertNotIn("肯定", fact)

    def test_not_live_returns_empty(self) -> None:
        card = make_card(gsi=gsi_payload(phase="freeze", seconds_left=15.0))
        self.assertEqual(build_candidates(card, self.graph), [])

    def test_dead_self_returns_empty(self) -> None:
        card = make_card(gsi=gsi_payload(health=0))
        self.assertEqual(build_candidates(card, self.graph), [])

    def test_no_position_still_returns_wait(self) -> None:
        card = make_card(radar=radar_payload(teammates=[], enemies=[]))
        card["self"]["x"] = None
        card["self"]["y"] = None
        card["self"]["area"] = None
        candidates = build_candidates(card, self.graph)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["action"], "WAIT")


if __name__ == "__main__":
    unittest.main()
