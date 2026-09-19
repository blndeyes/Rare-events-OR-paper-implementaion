"""Run official videojedi JEDi in the interpreter selected by --jedi-python."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from or_video_reproduction.evaluation.finegrained.external import ModelUnavailable


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _write(args.output, {"status": "failed", "reason": f"cannot read request: {error}"})
        return 1
    try:
        from or_video_reproduction.evaluation.finegrained.official import compute_jedi_inprocess

        payload = compute_jedi_inprocess(
            [Path(path) for path in request["reference_videos"]],
            [Path(path) for path in request["generated_videos"]],
            feature_path=Path(request["feature_path"]),
            model_dir=Path(request["model_dir"]),
            config_path=Path(request["config_path"]),
        )
    except ModelUnavailable as error:
        _write(args.output, {"status": error.status, "reason": error.reason})
        return 1
    except Exception as error:  # noqa: BLE001
        _write(args.output, {"status": "failed", "reason": str(error)})
        return 1
    _write(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
