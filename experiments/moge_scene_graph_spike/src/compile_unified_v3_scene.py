"""CLI for clean unified V3 compilation and its separate regression audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .unified_v3_scene_compiler import DEFAULT_OUTPUT, compile_clean, finalize_handoff, run_regression_audit


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("clean_reconstruction","regression_audit"),default="clean_reconstruction")
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if args.mode=="clean_reconstruction":
        result=compile_clean(args.output_dir.resolve());finalize_handoff(args.output_dir.resolve())
        print(json.dumps({"mode":args.mode,"output":str(args.output_dir),"objects":result["semantic_object_count"]},indent=2))
    else:
        result=run_regression_audit(DEFAULT_OUTPUT,args.output_dir.resolve())
        print(json.dumps({"mode":args.mode,"output":str(args.output_dir),"counts":result["counts"]},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
