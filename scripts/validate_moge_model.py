from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import moge_engine


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate configured local Microsoft MoGe settings through the isolated worker."
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Validate paths, imports, and device visibility without loading the checkpoint.",
    )
    args = parser.parse_args()

    try:
        config = moge_engine.validate_model_config()
        print("MoGe configuration:")
        for key in ("python", "repository", "checkpoint", "version", "device", "model", "allow_cpu"):
            print(f"  {key}: {config[key]}")

        result = moge_engine.validate_model(load_model=not args.skip_model_load)
    except moge_engine.MogeEngineError as exc:
        print(f"MoGe validation failed: {exc}", file=sys.stderr)
        return 1

    print("MoGe validation result:")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
