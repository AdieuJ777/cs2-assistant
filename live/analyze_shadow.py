"""Summarise a live shadow log.

The log contains rotation records and mid-round agent records:

- ``advice`` / ``outcome``: the v2 rotation call and what the player actually
  did ``--outcome-horizon`` seconds later (default 5 s);
- ``plan``: a mid-round plan selected by the template or the LLM;
- ``plan_llm``: one LLM attempt, including timeouts and validation failures;
- ``plan_outcome``: what the player did 3 s / 5 s after pressing the hotkey.

Run:
    python live/analyze_shadow.py live/shadow_log.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _is_rotate(action: str | None) -> bool:
    return bool(action) and str(action).startswith("ROTATE")


def _binary_report(records: list[dict], label: str) -> None:
    tp = sum(
        1
        for r in records
        if _is_rotate(r.get("advice_action")) and _is_rotate(r.get("actual_action"))
    )
    fp = sum(
        1
        for r in records
        if _is_rotate(r.get("advice_action")) and not _is_rotate(r.get("actual_action"))
    )
    fn = sum(
        1
        for r in records
        if not _is_rotate(r.get("advice_action")) and _is_rotate(r.get("actual_action"))
    )
    tn = sum(
        1
        for r in records
        if not _is_rotate(r.get("advice_action")) and not _is_rotate(r.get("actual_action"))
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_rotation_rate = fp / (fp + tn) if fp + tn else 0.0
    missed_rotation_rate = fn / (fn + tp) if fn + tp else 0.0
    print(
        f"{label}: precision={precision:.3f} recall={recall:.3f} "
        f"false_rotation={false_rotation_rate:.3f} "
        f"missed_rotation={missed_rotation_rate:.3f} "
        f"(tp={tp}, fp={fp}, tn={tn}, fn={fn})"
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def plan_report(records: list[dict]) -> dict:
    """Summarise mid-round plan, LLM-attempt and plan-outcome records.

    Returns the summary dict so tests and notebooks can use the numbers
    without parsing stdout.
    """
    plans = [r for r in records if r.get("record_type") == "plan"]
    attempts = [r for r in records if r.get("record_type") == "plan_llm"]
    outcomes = [r for r in records if r.get("record_type") == "plan_outcome"]
    summary: dict = {
        "plans": {"total": len(plans), "by_source": {}, "latency_s": []},
        "attempts": {"total": len(attempts), "ok": 0, "errors": {}, "validation": {}},
        "outcomes": {"total": len(outcomes), "status": {}, "adoption_by_horizon": {}},
    }
    if not (plans or attempts or outcomes):
        return summary

    if plans:
        by_source = Counter(r.get("source") for r in plans)
        summary["plans"]["by_source"] = dict(by_source)
        calls = [str(r.get("call") or "") for r in plans]
        facts = [len(r.get("grounded_facts") or []) for r in plans]
        latencies = [
            float(r["llm_latency_s"])
            for r in plans
            if r.get("llm_latency_s") is not None
        ]
        summary["plans"]["latency_s"] = latencies
        print(f"\nplan records: {len(plans)}  sources={dict(by_source)}")
        print(
            f"plan calls: max_chars={max((len(c) for c in calls), default=0)} "
            f"avg_facts={sum(facts) / len(facts):.1f}"
        )
        if latencies:
            print(
                f"LLM plan latency: p50={_percentile(latencies, 0.5):.2f}s "
                f"p95={_percentile(latencies, 0.95):.2f}s"
            )

    if attempts:
        errors = Counter(r.get("error") for r in attempts if r.get("error"))
        validation = Counter(r.get("validation") for r in attempts)
        ok = sum(1 for r in attempts if r.get("ok"))
        latencies = [
            float(r["latency_s"])
            for r in attempts
            if r.get("latency_s") is not None
        ]
        summary["attempts"].update(
            {"ok": ok, "errors": dict(errors), "validation": dict(validation)}
        )
        print(
            f"LLM attempts: ok={ok}/{len(attempts)} "
            f"({ok / len(attempts):.1%}) validation={dict(validation)}"
        )
        if errors:
            print("LLM errors:", dict(errors))
        if latencies:
            print(
                f"LLM attempt latency: p50={_percentile(latencies, 0.5):.2f}s "
                f"p95={_percentile(latencies, 0.95):.2f}s"
            )

    if not outcomes:
        return summary

    statuses = Counter(r.get("status") for r in outcomes)
    summary["outcomes"]["status"] = dict(statuses)
    print(f"\nplan outcomes: {len(outcomes)}  statuses={dict(statuses)}")

    by_horizon: dict[float, list[dict]] = defaultdict(list)
    for record in outcomes:
        # Adoption is only meaningful when the plan had a target area and the
        # player's area could be measured. Older logs may label these "ok"
        # even without a target, so filter on the fields as well.
        if (
            record.get("status") == "ok"
            and record.get("target_area")
            and record.get("actual_area")
        ):
            by_horizon[float(record.get("horizon_s", 0.0))].append(record)
    for horizon in sorted(by_horizon):
        records_for_horizon = by_horizon[horizon]
        adopted = sum(1 for r in records_for_horizon if r.get("adopted"))
        rate = adopted / len(records_for_horizon)
        summary["outcomes"]["adoption_by_horizon"][horizon] = {
            "adopted": adopted,
            "n": len(records_for_horizon),
            "rate": round(rate, 3),
        }
        print(
            f"adoption @{horizon:g}s: {adopted}/{len(records_for_horizon)} "
            f"({rate:.1%})"
        )
        by_source: dict[str, list[dict]] = defaultdict(list)
        for record in records_for_horizon:
            by_source[str(record.get("plan_source"))].append(record)
        for source, source_records in sorted(by_source.items()):
            source_adopted = sum(1 for r in source_records if r.get("adopted"))
            print(
                f"  {source:>8}: {source_adopted}/{len(source_records)} "
                f"({source_adopted / len(source_records):.1%})"
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("live/shadow_log.jsonl"))
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"no shadow log at {args.path}")
    records = [
        json.loads(line)
        for line in args.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("shadow log is empty")

    advice = [r for r in records if r.get("record_type", "advice") == "advice"]
    outcomes = [r for r in records if r.get("record_type") == "outcome"]

    plan_count = sum(1 for r in records if r.get("record_type") == "plan")
    llm_count = sum(1 for r in records if r.get("record_type") == "plan_llm")
    plan_outcome_count = sum(
        1 for r in records if r.get("record_type") == "plan_outcome"
    )
    print(
        f"records: {len(records)}  (advice={len(advice)}, outcome={len(outcomes)}, "
        f"plan={plan_count}, plan_llm={llm_count}, "
        f"plan_outcome={plan_outcome_count})"
    )

    if advice:
        actions = Counter(r.get("action") for r in advice)
        confidence = Counter(r.get("confidence") for r in advice)
        rounds = {(r.get("match_map"), r.get("round")) for r in advice}
        rotations = sum(
            count for action, count in actions.items() if _is_rotate(action)
        )
        destinations = Counter(
            r.get("destination") for r in advice if r.get("destination")
        )
        warnings = Counter(
            warning for r in advice for warning in r.get("warnings", [])
        )
        print(f"advice rounds: {len(rounds)}")
        print(
            f"rotation calls: {rotations} "
            f"({rotations / len(advice):.1%} of advice)"
        )
        print("advice actions:", dict(actions))
        print("advice confidence:", dict(confidence))
        print("destinations:", dict(destinations))
        if warnings:
            print("advice warnings:")
            for warning, count in warnings.most_common():
                print(f"  {count:>4}  {warning}")

        feature_rows = [
            record["features"]
            for record in advice
            if isinstance(record.get("features"), dict)
        ]
        if feature_rows:
            print("live feature means:")
            for key in (
                "ct_alive_count",
                "enemy_alive_count",
                "enemy_visible_count",
                "enemy_unknown_count",
                "enemy_visible_ratio",
                "enemy_nearest_distance",
                "team_health_mean",
            ):
                values = [
                    float(row[key])
                    for row in feature_rows
                    if row.get(key) is not None
                ]
                if values:
                    print(f"  {key:>24}  {sum(values) / len(values):.3f}")
            areas = Counter(row.get("area") for row in feature_rows)
            print(f"  {'self area':>24}  {dict(areas)}")

    plan_report(records)

    if not outcomes:
        print(
            "\nno rotation outcome records yet; "
            "play a live round with the server running."
        )
        return

    statuses = Counter(r.get("status") for r in outcomes)
    valid = [r for r in outcomes if r.get("status") == "ok"]
    print("\noutcome statuses:", dict(statuses))
    print(f"outcomes with a valid 5 s action: {len(valid)}")
    if not valid:
        return

    matched = sum(1 for r in valid if r.get("matched"))
    print(f"exact advice match: {matched}/{len(valid)} ({matched / len(valid):.1%})")

    _binary_report(valid, "binary rotate (all valid)")

    spot = [
        r for r in valid if (r.get("advice_newly_visible_count") or 0) > 0
    ]
    if spot:
        print(f"\nspot-event outcomes: {len(spot)}")
        _binary_report(spot, "binary rotate (spot events)")
        spot_match = sum(1 for r in spot if r.get("matched"))
        print(
            f"spot-event exact match: {spot_match}/{len(spot)} "
            f"({spot_match / len(spot):.1%})"
        )
    else:
        print("\nno spot-event outcomes recorded (newly_visible_count unavailable).")

    by_advice: dict[str, list[bool]] = defaultdict(list)
    by_confidence: dict[str, list[bool]] = defaultdict(list)
    for record in valid:
        by_advice[str(record.get("advice_action"))].append(bool(record.get("matched")))
        by_confidence[str(record.get("advice_confidence"))].append(
            bool(record.get("matched"))
        )
    print("match rate by advice:")
    for action, values in sorted(by_advice.items()):
        print(f"  {action:>10}  {sum(values)}/{len(values)} ({sum(values) / len(values):.1%})")
    print("match rate by confidence:")
    for label, values in sorted(by_confidence.items()):
        print(f"  {label:>10}  {sum(values)}/{len(values)} ({sum(values) / len(values):.1%})")


if __name__ == "__main__":
    main()
