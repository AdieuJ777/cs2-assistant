"""Tests for the mid-round plan cache and fingerprint."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.midround.plan_cache import PlanCache, fingerprint  # noqa: E402


def card(
    *,
    round_number: int = 5,
    seconds: float = 77.0,
    self_area: str = "Mid",
    health: int = 100,
    enemies: list | None = None,
    teammates: list | None = None,
    bomb: str = "carried",
    model_action: str = "ROTATE_B",
    phase: str = "live",
) -> dict:
    return {
        "round": {
            "number": round_number,
            "phase": phase,
            "seconds_since_freeze": seconds,
            "time_left": 115.0 - seconds,
        },
        "self": {"area": self_area, "health": health},
        "team": {
            "members": teammates
            if teammates is not None
            else [
                {"area": "B", "health": 80, "alive": True},
                {"area": "A", "health": 45, "alive": True},
            ]
        },
        "enemies": {
            "visible": enemies
            if enemies is not None
            else [{"area": "A", "distance": 1000.0}]
        },
        "bomb": {"state": bomb},
        "model": {"action": model_action},
    }


def plan(created_at: float = 1000.0, **overrides) -> dict:
    payload = {
        "plan_id": "5:15:safe:template",
        "created_at": created_at,
        "fingerprint": fingerprint(card()),
        "source": "template",
        "selected_id": "safe",
        "call": "转 B，走安全路线",
    }
    payload.update(overrides)
    return payload


class FingerprintTests(unittest.TestCase):
    def test_stable_and_serialisable(self) -> None:
        first = fingerprint(card())
        second = fingerprint(card())
        self.assertEqual(first, second)
        self.assertEqual(first["time_bucket"], 15)
        self.assertEqual(first["visible_enemies"], ["A"])
        self.assertEqual(first["teammates"], ["A:45", "B:80"])

    def test_changes_with_state(self) -> None:
        base = fingerprint(card())
        self.assertNotEqual(base, fingerprint(card(round_number=6)))
        self.assertNotEqual(base, fingerprint(card(seconds=82.0)))
        self.assertNotEqual(base, fingerprint(card(health=25)))
        self.assertNotEqual(base, fingerprint(card(self_area="B")))
        self.assertNotEqual(base, fingerprint(card(bomb="planted")))
        self.assertNotEqual(base, fingerprint(card(model_action="HOLD")))
        self.assertNotEqual(base, fingerprint(card(phase="freeze")))

    def test_visible_enemy_order_does_not_matter(self) -> None:
        first = fingerprint(card(enemies=[{"area": "A"}, {"area": "B"}]))
        second = fingerprint(card(enemies=[{"area": "B"}, {"area": "A"}]))
        self.assertEqual(first, second)


class PlanCacheTests(unittest.TestCase):
    def test_fresh_hit(self) -> None:
        cache = PlanCache()
        cache.put(plan())
        cached, age = cache.get(fingerprint(card()), now=1001.0)
        self.assertIsNotNone(cached)
        self.assertAlmostEqual(age, 1.0)
        self.assertEqual(cached["selected_id"], "safe")

    def test_stale_still_returned_before_max_age(self) -> None:
        cache = PlanCache()
        cache.put(plan())
        cached, age = cache.get(fingerprint(card()), now=1003.5)
        self.assertIsNotNone(cached)
        self.assertAlmostEqual(age, 3.5)

    def test_expired_is_dropped(self) -> None:
        cache = PlanCache()
        cache.put(plan())
        cached, age = cache.get(fingerprint(card()), now=1005.0)
        self.assertIsNone(cached)
        self.assertAlmostEqual(age, 5.0)
        self.assertFalse(cache.status()["has_plan"])

    def test_fingerprint_mismatch_invalidates(self) -> None:
        cache = PlanCache()
        cache.put(plan())
        cached, _ = cache.get(fingerprint(card(round_number=6)), now=1000.5)
        self.assertIsNone(cached)

    def test_latest_ignores_fingerprint(self) -> None:
        cache = PlanCache()
        cache.put(plan())
        cached, age = cache.latest(now=1001.0)
        self.assertIsNotNone(cached)
        self.assertAlmostEqual(age, 1.0)

    def test_invalidate(self) -> None:
        cache = PlanCache()
        cache.put(plan())
        cache.invalidate()
        self.assertIsNone(cache.latest(now=1000.5)[0])

    def test_concurrent_access_is_safe(self) -> None:
        cache = PlanCache()
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                for step in range(50):
                    cache.put(plan(created_at=1000.0 + index))
                    cache.get(fingerprint(card()), now=1000.5 + index)
                    cache.status()
            except BaseException as exc:  # pragma: no cover - diagnostic
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
