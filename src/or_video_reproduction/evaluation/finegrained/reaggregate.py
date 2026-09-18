"""Recompute anatomy/control summaries from saved per-clip results.

This path does not decode videos or load DETR, ViTPose, Depth Anything, CLIP, or
DINO. Unrelated completed metrics are copied unchanged. Malformed per-clip input
is refused rather than filled in.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .control import reaggregate_control_clip, summarize_control_group
from .protocol import FROZEN_MODELS, SCHEMA_VERSION, WAN_DIAGNOSTIC_WARNING, write_json
from .scoring import aggregate_clip_detection_rates


PRESERVED_METRIC_KEYS = (
    "clip_cmmd",
    "clip_kid",
    "dinov2",
    "dinov3",
    "hands",
    "fvmd",
    "jedi",
)


class ReaggregationError(ValueError):
    """Saved per-clip results are insufficient or malformed."""


def _require_mapping(value: object, *, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReaggregationError(f"{what} must be a JSON object")
    return value


def reaggregate_person_detection(person: Mapping[str, Any]) -> dict[str, Any]:
    per_clip = person.get("per_clip")
    if not isinstance(per_clip, list) or not per_clip:
        raise ReaggregationError("anatomy person_detection is missing per_clip records")
    updated = dict(person)
    updated["real"] = aggregate_clip_detection_rates(per_clip, "real")
    updated["generated"] = aggregate_clip_detection_rates(per_clip, "generated")
    updated["status"] = person.get("status", "ok")
    return updated


def reaggregate_anatomy(anatomy: Mapping[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(dict(anatomy))
    parts = updated.get("parts")
    if not isinstance(parts, dict):
        raise ReaggregationError("anatomy is missing parts")
    person = parts.get("person_detection")
    if person is None:
        raise ReaggregationError("anatomy is missing person_detection")
    if isinstance(person, Mapping) and person.get("status") in {"unavailable", "blocked", "failed"}:
        return updated
    parts["person_detection"] = reaggregate_person_detection(_require_mapping(person, what="person_detection"))
    return updated


def reaggregate_control(control: Mapping[str, Any]) -> dict[str, Any]:
    if control.get("status") in {"unavailable", "blocked", "failed"}:
        updated = dict(control)
        updated.setdefault("recovery_scope", "person_only_plus_independent_depth")
        return updated
    per_clip = control.get("per_clip")
    if not isinstance(per_clip, list) or not per_clip:
        raise ReaggregationError("control is missing per_clip records")
    rewritten = [reaggregate_control_clip(_require_mapping(row, what="control per-clip")) for row in per_clip]
    summary = summarize_control_group(rewritten)
    payload = dict(control)
    payload.update(summary)
    payload["per_clip"] = rewritten
    if control.get("independent_depth_model") is not None:
        payload["independent_depth_model"] = copy.deepcopy(control["independent_depth_model"])
    return payload


def reaggregate_group(record: Mapping[str, Any]) -> dict[str, Any]:
    group = copy.deepcopy(dict(record))
    metrics = _require_mapping(group.get("metrics"), what="group metrics")
    if "anatomy" in metrics:
        metrics["anatomy"] = reaggregate_anatomy(_require_mapping(metrics["anatomy"], what="anatomy"))
    if "control" in metrics:
        metrics["control"] = reaggregate_control(_require_mapping(metrics["control"], what="control"))
    group["metrics"] = metrics
    return group


def load_per_clip_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReaggregationError(f"per-clip results not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReaggregationError(f"malformed per-clip results: {error}") from error
    return _require_mapping(payload, what="per-clip results")


def preserved_metric_fingerprint(results: Mapping[str, Any]) -> dict[str, Any]:
    fingerprint: dict[str, Any] = {}
    for group_id, record in results.items():
        metrics = record.get("metrics") if isinstance(record, Mapping) else {}
        if not isinstance(metrics, Mapping):
            continue
        fingerprint[str(group_id)] = {
            name: copy.deepcopy(metrics[name]) for name in PRESERVED_METRIC_KEYS if name in metrics
        }
    return fingerprint


def assert_preserved_metrics(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    left = preserved_metric_fingerprint(before)
    right = preserved_metric_fingerprint(after)
    if left != right:
        raise ReaggregationError("aggregate-only processing changed unrelated completed metrics")


def _source_summary_fields(source_root: Path, results: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = source_root / "evaluation-manifest.json"
    groups = []
    pairing_rule = None
    wan_in = None
    wan_note = None
    input_root = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ReaggregationError(f"malformed evaluation-manifest.json: {error}") from error
        if isinstance(manifest, Mapping):
            pairing_rule = manifest.get("pairing_rule")
            wan_in = manifest.get("wan_in_frozen_4dor_split")
            wan_note = manifest.get("wan_note")
            input_root = manifest.get("input_root")
            if isinstance(manifest.get("groups"), list):
                groups = manifest["groups"]
    if not groups:
        groups = [
            {
                "id": group_id,
                "comparison_class": record.get("comparison_class") if isinstance(record, Mapping) else None,
                "sample_count": None,
                "clip_ids": [],
            }
            for group_id, record in results.items()
        ]
    return {
        "input_root": input_root,
        "pairing_rule": pairing_rule or "See the original evaluation-manifest for pairing.",
        "wan_in_frozen_4dor_split": wan_in,
        "wan_note": wan_note or WAN_DIAGNOSTIC_WARNING,
        "groups": groups,
    }


def flatten_metric_status(results: dict[str, dict[str, object]]) -> dict[str, object]:
    status: dict[str, object] = {}
    for group_id, group in results.items():
        metrics = group.get("metrics") if isinstance(group, Mapping) else {}
        if not isinstance(metrics, Mapping):
            continue
        for name, record in metrics.items():
            key = f"{group_id}:{name}"
            if isinstance(record, dict):
                status[key] = {
                    "status": record.get("status", "ok"),
                    "reason": record.get("reason"),
                    "diagnostic_only": record.get("diagnostic_only"),
                    "role": record.get("role"),
                }
            else:
                status[key] = {"status": "ok"}
    return status


def write_reaggregated_summaries(
    *,
    source_root: Path,
    output_root: Path,
    results: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    from .runner import render_summary

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "per-clip-results.json", results)
    metric_status = flatten_metric_status(dict(results))
    write_json(output_root / "metric-status.json", metric_status)
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "groups": {
            group_id: {
                "comparison_class": record.get("comparison_class") if isinstance(record, Mapping) else None,
                "metrics": record.get("metrics") if isinstance(record, Mapping) else None,
            }
            for group_id, record in results.items()
        },
        "do_not_merge_mmor_and_4dor": True,
        "cross_method_benchmark_complete": False,
        "reason_incomplete": (
            "WAN is a single take-9 diagnostic clip and is not on the frozen six-clip 4D-OR split"
        ),
        "reaggregation": provenance,
    }
    write_json(output_root / "aggregate-results.json", aggregate)
    fields = _source_summary_fields(source_root, results)
    summary_payload = {
        **fields,
        "output_root": str(output_root),
        "source_results": str(source_root),
        "results": results,
        "metric_status": metric_status,
        "models": FROZEN_MODELS,
        "reaggregation": provenance,
    }
    (output_root / "results-summary.md").write_text(render_summary(summary_payload), encoding="utf-8")
    write_json(output_root / "reaggregation-provenance.json", provenance)
    return summary_payload


def run_reaggregation(
    *,
    source_root: Path,
    output_root: Path,
    source_commit: str | None = None,
    reaggregation_commit: str | None = None,
) -> dict[str, Any]:
    if source_root.resolve() == output_root.resolve():
        raise ReaggregationError("refuse to overwrite the source results directory")
    original = load_per_clip_results(source_root / "per-clip-results.json")
    before = copy.deepcopy(original)
    results = {group_id: reaggregate_group(_require_mapping(record, what=f"group {group_id}")) for group_id, record in original.items()}
    assert_preserved_metrics(before, results)
    provenance = {
        "mode": "aggregate_only",
        "source_results": str(source_root),
        "source_per_clip_results": str(source_root / "per-clip-results.json"),
        "source_commit": source_commit,
        "reaggregation_commit": reaggregation_commit,
        "models_loaded": [],
        "recomputed": ["anatomy.person_detection.group_rates", "control.person_only_scope"],
        "preserved": list(PRESERVED_METRIC_KEYS),
    }
    return write_reaggregated_summaries(
        source_root=source_root,
        output_root=output_root,
        results=results,
        provenance=provenance,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", type=str)
    parser.add_argument("--reaggregation-commit", type=str)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_reaggregation(
        source_root=args.source_results,
        output_root=args.output_root,
        source_commit=args.source_commit,
        reaggregation_commit=args.reaggregation_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
