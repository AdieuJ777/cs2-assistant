"""Pure-stdlib inference for the exported two-stage rotation policy.

The artifact in ``live/models/rotation_policy.json`` is produced by
``benchmarks/enemy_visibility/export_event_policy.py``. This module implements
the same random-forest traversal without scikit-learn, plus the conversion from
a live game state to the model's feature row.

Safety rule: a rotation recommendation is only produced when at least one enemy
is visible in the radar input. Without radar, the advisor returns ``NO_RADAR``
instead of guessing from CT-only state, because the model was trained with
enemy features present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

AREAS = ("A", "B", "Mid", "CT")
ACTION_LABELS = ["HOLD", "ROTATE_A", "ROTATE_B", "ROTATE_MID", "ROTATE_CT"]


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def nearest_area(centroids: dict[str, list[float]], x: float, y: float) -> str:
    return min(
        centroids,
        key=lambda area: math.hypot(x - centroids[area][0], y - centroids[area][1]),
    )


class RotationPolicy:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.cat_features: list[str] = artifact["cat_features"]
        self.num_features: list[str] = artifact["num_features"]
        self.feature_order: list[str] = artifact["feature_order"]
        self.cat_index = {
            column: {value: index for index, value in enumerate(values)}
            for column, values in artifact["cat_mappings"].items()
        }
        self.num_medians: dict[str, float] = artifact["num_medians"]
        self.centroids: dict[str, list[float]] = artifact["area_centroids"]
        self.threshold = float(artifact["threshold"])
        self.stage1 = artifact["stage1"]
        self.stage2 = artifact["stage2"]

    @classmethod
    def load(cls, path: Path | str) -> "RotationPolicy":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ trees
    @staticmethod
    def _tree_proba(tree: dict[str, Any], values: list[float]) -> list[float]:
        node = 0
        left = tree["left"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        while left[node] != -1:
            node = (
                left[node]
                if values[feature[node]] <= threshold[node]
                else tree["right"][node]
            )
        leaf = tree["value"][node]
        total = sum(leaf) or 1.0
        return [value / total for value in leaf]

    def _forest_proba(
        self, forest: list[dict[str, Any]], values: list[float], n_classes: int
    ) -> list[float]:
        totals = [0.0] * n_classes
        for tree in forest:
            leaf = self._tree_proba(tree, values)
            for index, value in enumerate(leaf):
                totals[index] += value
        return [value / len(forest) for value in totals]

    # --------------------------------------------------------------- encoding
    def encode(self, features: dict[str, Any]) -> list[float]:
        encoded: list[float] = []
        for column in self.feature_order:
            value = features.get(column)
            if column in self.cat_features:
                mapping = self.cat_index[column]
                text = str(value)
                code = mapping.get(text)
                if code is None:
                    code = mapping.get("other", 0)
                encoded.append(float(code))
            else:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = self.num_medians[column]
                if math.isnan(number):
                    number = self.num_medians[column]
                encoded.append(number)
        return encoded

    # -------------------------------------------------------------- inference
    def predict_encoded(self, encoded: dict[str, float]) -> dict[str, Any]:
        values = [float(encoded[column]) for column in self.feature_order]
        return self._predict_values(values)

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        return self._predict_values(self.encode(features))

    def _predict_values(self, values: list[float]) -> dict[str, Any]:
        stage1_proba = self._forest_proba(
            self.stage1["trees"], values, len(self.stage1["classes"])
        )
        rotate_index = self.stage1["classes"].index(1)
        rotate_proba = stage1_proba[rotate_index]

        stage2_proba = self._forest_proba(
            self.stage2["trees"], values, len(self.stage2["classes"])
        )
        best = max(range(len(stage2_proba)), key=stage2_proba.__getitem__)
        destination_index = int(self.stage2["classes"][best])
        destination = ACTION_LABELS[destination_index]
        rotate = rotate_proba >= self.threshold

        return {
            "stage1_proba": rotate_proba,
            "rotate": rotate,
            "action": destination if rotate else "HOLD",
            "destination": destination if rotate else None,
            "destination_argmax": destination,
            "destination_proba": float(stage2_proba[best]),
            "threshold": self.threshold,
        }


def state_to_features(artifact: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Convert a live state to the model's raw feature dict.

    ``teammates`` are the *other* alive CTs (self is added separately).
    ``enemies`` are visible enemies only; each may carry ``area`` or raw x/y.
    """
    centroids: dict[str, list[float]] = artifact["area_centroids"]
    self_state = state["self"]
    teammates = [mate for mate in state.get("teammates", []) if mate.get("health", 1) > 0]
    enemies = [
        enemy
        for enemy in state.get("enemies", [])
        if enemy.get("visible", True) and enemy.get("health", 1) > 0
    ]

    self_x = float(self_state["x"])
    self_y = float(self_state["y"])
    self_area = str(self_state.get("area") or nearest_area(centroids, self_x, self_y))
    self_health = float(self_state.get("health", 100))
    self_armor = float(self_state.get("armor", 0))
    self_equip = float(self_state.get("equip", 0))
    self_yaw = float(self_state.get("yaw", 0.0))

    counts = {area: 0 for area in AREAS}
    if self_area in counts:
        counts[self_area] += 1
    teammate_health = [self_health]
    team_positions = [(self_x, self_y)]
    for mate in teammates:
        mate_x = float(mate["x"])
        mate_y = float(mate["y"])
        mate_area = str(mate.get("area") or nearest_area(centroids, mate_x, mate_y))
        if mate_area in counts:
            counts[mate_area] += 1
        teammate_health.append(float(mate.get("health", 100)))
        team_positions.append((mate_x, mate_y))

    distances = {
        area: math.hypot(self_x - centre[0], self_y - centre[1]) / 1000.0
        for area, centre in centroids.items()
        if area in AREAS
    }
    nearest = min(distances, key=distances.get)

    enemy_positions: list[tuple[float, float]] = []
    enemy_areas: list[str] = []
    enemy_distances: list[float] = []
    for enemy in enemies:
        enemy_x = float(enemy["x"])
        enemy_y = float(enemy["y"])
        enemy_area = str(enemy.get("area") or nearest_area(centroids, enemy_x, enemy_y))
        enemy_positions.append((enemy_x, enemy_y))
        enemy_areas.append(enemy_area)
        enemy_distances.append(math.hypot(self_x - enemy_x, self_y - enemy_y))

    if not enemy_positions:
        raise ValueError("state_to_features requires at least one visible enemy")
    nearest_enemy = min(range(len(enemy_positions)), key=enemy_distances.__getitem__)
    enemy_x, enemy_y = enemy_positions[nearest_enemy]
    bearing = math.degrees(math.atan2(enemy_y - self_y, enemy_x - self_x))

    enemy_counts = {area: 0 for area in AREAS}
    enemy_counts["other"] = 0
    for area in enemy_areas:
        if area in enemy_counts:
            enemy_counts[area] += 1
        else:
            enemy_counts["other"] += 1

    team_nearest = min(
        math.hypot(team_x - enemy_x, team_y - enemy_y)
        for team_x, team_y in team_positions
        for enemy_x, enemy_y in enemy_positions
    )

    visible_count = len(enemies)
    enemy_alive_value = state.get("enemy_alive_count")
    if enemy_alive_value is None:
        enemy_alive_value = visible_count + int(state.get("enemy_unknown_count", 0) or 0)
    enemy_alive_count = int(enemy_alive_value)
    unknown_count = max(enemy_alive_count - visible_count, 0)
    pistol = int(int(state.get("round_num", 1)) in {1, 13})

    return {
        "area": self_area,
        "round_phase": state.get("round_phase", "first_half"),
        "prev_round_winner_side": state.get("prev_round_winner_side", "none"),
        "prev_round_attack_site": state.get("prev_round_attack_site", "unknown"),
        "nearest_area": nearest,
        "health": self_health,
        "armor": self_armor,
        "current_equip_value": self_equip,
        "count_A": counts["A"],
        "count_B": counts["B"],
        "count_Mid": counts["Mid"],
        "count_CT": counts["CT"],
        "dist_A": distances["A"],
        "dist_B": distances["B"],
        "dist_Mid": distances["Mid"],
        "dist_CT": distances["CT"],
        "team_health_mean": sum(teammate_health) / len(teammate_health),
        "score_diff_ct_minus_t_before": float(state.get("score_diff", 0)),
        "round_num": int(state.get("round_num", 1)),
        "pistol_round_flag": pistol,
        "ct_alive_count": len(teammates) + 1,
        "seconds_since_freeze": float(state.get("seconds_since_freeze", 0.0)),
        "seconds_since_previous_event": float(
            state.get("seconds_since_previous_event", 99.0)
        ),
        "enemy_alive_count": enemy_alive_count,
        "enemy_visible_count": visible_count,
        "enemy_unknown_count": unknown_count,
        "enemy_visible_ratio": visible_count / enemy_alive_count
        if enemy_alive_count
        else 0.0,
        "newly_visible_count": int(state.get("newly_visible_count", visible_count)),
        "enemy_nearest_distance": enemy_distances[nearest_enemy],
        "enemy_mean_distance": sum(enemy_distances) / len(enemy_distances),
        "enemy_nearest_area": enemy_areas[nearest_enemy],
        "enemy_nearest_relative_bearing": wrap_degrees(bearing - self_yaw),
        "enemy_team_nearest_distance": team_nearest,
        "self_spots_enemy": int(bool(state.get("self_spots_enemy", False))),
        "self_spot_count": int(
            state.get("self_spot_count", int(bool(state.get("self_spots_enemy", False))))
        ),
        **{f"enemy_visible_{area}": enemy_counts[area] for area in (*AREAS, "other")},
    }


class RotationAdvisor:
    """Live advisor with an explicit radar requirement."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.policy = RotationPolicy(artifact)

    @classmethod
    def load(cls, path: Path | str) -> "RotationAdvisor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def advise(self, state: dict[str, Any]) -> dict[str, Any]:
        warnings: list[str] = []
        map_name = str(state.get("map", "")).lower()
        if "dust2" not in map_name:
            return self._no_advice("NOT_DUST2", "Rotation policy is Dust2-only.", warnings)
        if str(state.get("side", "ct")).lower() != "ct":
            return self._no_advice("NOT_CT", "Rotation policy is CT-only.", warnings)
        if state.get("phase") == "freeze":
            return self._no_advice(
                "BUY_PHASE", "Freeze time: use the buy/save advice.", warnings
            )
        if state.get("phase", "live") != "live":
            return self._no_advice(
                "NO_MID_ROUND", "No live round in progress.", warnings
            )

        enemies = [
            enemy
            for enemy in state.get("enemies", [])
            if enemy.get("visible", True) and enemy.get("health", 1) > 0
        ]
        if not enemies:
            warnings.append(
                "No visible enemy: radar input is required before a rotation call."
            )
            return self._no_advice(
                "NO_RADAR",
                "No visible enemy in the radar feed; not guessing from CT state.",
                warnings,
            )
        if state.get("radar_stale"):
            warnings.append("Radar data is stale (>1.5 s); treat the call as degraded.")
        if state.get("teammates_perception", True) is False:
            warnings.append(
                "Teammate positions are missing; CT counts would be out of "
                "distribution, so no rotation call is made."
            )
            return self._no_advice(
                "NO_TEAMMATE_DATA",
                "Teammate radar data is required for a safe rotation call.",
                warnings,
            )

        features = state_to_features(self.artifact, state)
        prediction = self.policy.predict(features)
        confidence = (
            prediction["stage1_proba"]
            if prediction["rotate"]
            else 1.0 - prediction["stage1_proba"]
        )
        if confidence >= 0.75:
            label = "high"
        elif confidence >= 0.60:
            label = "medium"
        else:
            label = "low"

        nearest_area = features["enemy_nearest_area"]
        if prediction["rotate"]:
            reason = (
                f"Visible T at {nearest_area} "
                f"({features['enemy_nearest_distance']:.0f}u); "
                f"P(rotate)={prediction['stage1_proba']:.2f}."
            )
        else:
            reason = (
                f"No rotation evidence (P(rotate)={prediction['stage1_proba']:.2f}); "
                f"visible T at {nearest_area}."
            )

        return {
            "action": prediction["action"],
            "destination": prediction["destination"],
            "confidence": label,
            "confidence_value": confidence,
            "stage1_proba": prediction["stage1_proba"],
            "destination_proba": prediction["destination_proba"],
            "radar_available": True,
            "reason": reason,
            "posture": "ROTATE" if prediction["rotate"] else "HOLD_ANGLE",
            "warnings": warnings,
            "features": features,
        }

    @staticmethod
    def _no_advice(
        action: str, reason: str, warnings: list[str]
    ) -> dict[str, Any]:
        return {
            "action": action,
            "destination": None,
            "confidence": "none",
            "confidence_value": None,
            "stage1_proba": None,
            "destination_proba": None,
            "radar_available": False,
            "reason": reason,
            "posture": "NONE",
            "warnings": warnings,
            "features": None,
        }
