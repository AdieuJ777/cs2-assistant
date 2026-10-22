from __future__ import annotations

import unittest

from live.advisor import RoundHistory, advise


def make_payload(
    *,
    round_num: int = 5,
    phase: str = "freezetime",
    money: int = 5000,
    team: str = "CT",
    armor: int = 0,
    helmet: bool = False,
    defuse: bool = False,
    weapons: dict | None = None,
    win_team: str | None = None,
    map_name: str = "de_dust2",
) -> dict:
    return {
        "map": {
            "name": map_name,
            "round": round_num,
            "team_ct": {"score": 3},
            "team_t": {"score": 2},
        },
        "round": {"phase": phase, "win_team": win_team, "bomb": "carried"},
        "player": {
            "team": team,
            "state": {
                "money": money,
                "health": 100,
                "armor": armor,
                "helmet": helmet,
                "defusekit": defuse,
            },
            "weapons": weapons or {},
        },
        "phase_countdowns": {"phase_ends_in": "15.0"},
    }


class AdvisorTests(unittest.TestCase):
    def test_full_buy(self) -> None:
        result = advise(make_payload(money=5000))
        self.assertEqual(result["recommendation"], "FULL_BUY")
        names = {item["name"] for item in result["shopping_list"]}
        self.assertIn("M4A1-S", names)
        self.assertIn("Kevlar + Helmet", names)
        self.assertIn("Defuse Kit", names)
        self.assertGreater(result["estimated_cost"], 0)
        self.assertGreaterEqual(result["reserve"], 0)

    def test_save(self) -> None:
        result = advise(make_payload(money=1200))
        self.assertEqual(result["recommendation"], "SAVE")
        self.assertEqual(result["shopping_list"], [])

    def test_pistol_round_does_not_buy_rifle(self) -> None:
        result = advise(make_payload(round_num=1, money=1000))
        self.assertEqual(result["recommendation"], "PISTOL_BUY")
        names = {item["name"] for item in result["shopping_list"]}
        self.assertNotIn("M4A1-S", names)
        self.assertIn("Kevlar + Helmet", names)
        self.assertLessEqual(result["estimated_cost"], 1000)

    def test_existing_rifle_only_needs_armor_and_utility(self) -> None:
        result = advise(
            make_payload(
                money=1500,
                armor=100,
                helmet=True,
                defuse=True,
                weapons={"weapon_0": {"name": "weapon_m4a1_silencer"}},
            )
        )
        self.assertEqual(result["recommendation"], "FULL_BUY")
        names = {item["name"] for item in result["shopping_list"]}
        self.assertNotIn("M4A1-S", names)

    def test_live_phase_stops_buy_advice(self) -> None:
        result = advise(make_payload(phase="live", money=5000))
        self.assertEqual(result["recommendation"], "BUY_PHASE_OVER")

    def test_t_side_is_not_supported(self) -> None:
        result = advise(make_payload(team="T", money=5000))
        self.assertEqual(result["recommendation"], "NOT_SUPPORTED")

    def test_history_and_enemy_prior(self) -> None:
        history = RoundHistory()
        history.update(make_payload(round_num=1, phase="over", win_team="ct"))
        history.update(make_payload(round_num=2, phase="over", win_team="ct"))
        self.assertEqual(history.t_loss_streak(), 2)
        result = advise(make_payload(round_num=3, money=3000), history)
        self.assertEqual(result["enemy_prior"]["label"], "eco or force")
        self.assertEqual(result["enemy_prior"]["confidence"], "low")

    def test_other_map_warns(self) -> None:
        result = advise(make_payload(map_name="de_mirage"))
        self.assertTrue(any("tuned for Dust2" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
