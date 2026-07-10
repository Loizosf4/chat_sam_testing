from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import sam_engine


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate configured Meta SAM 1 settings and load the model once."
    )
    parser.add_argument("--checkpoint", help="Override SAM_CHECKPOINT for this run.")
    parser.add_argument("--model-type", help="Override SAM_MODEL_TYPE for this run.")
    parser.add_argument("--device", help="Override SAM_DEVICE for this run.")
    args = parser.parse_args()

    try:
        config = sam_engine.validate_model_config(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            device=args.device,
        )
        print("SAM configuration is valid:")
        for key in ("checkpoint_path", "model_type", "device", "python_executable"):
            print(f"  {key}: {config[key]}")

        status = sam_engine.load_model(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            device=args.device,
        )
    except sam_engine.SamEngineError as exc:
        print(f"SAM validation failed: {exc}", file=sys.stderr)
        return 1

    print("SAM model loaded:")
    for key in ("model_type", "device", "checkpoint_path"):
        print(f"  {key}: {status[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
