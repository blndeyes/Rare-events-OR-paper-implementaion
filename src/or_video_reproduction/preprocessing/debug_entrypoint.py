"""Run a Python script with periodic stack dumps for long silent startup phases."""

from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path
import runpy
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-after", type=int, default=120)
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.dump_after <= 0:
        parser.error("--dump-after must be positive")

    faulthandler.enable()
    faulthandler.dump_traceback_later(args.dump_after, repeat=True)
    sys.argv = [str(args.script), *args.script_args]
    runpy.run_path(str(args.script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
