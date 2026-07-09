"""CLI for generic clean Unified V3 compilation and fixture regression audit."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .unified_v3_scene_compiler import (
    DEFAULT_MOGE,
    DEFAULT_OUTPUT,
    DEFAULT_SAM_DIR,
    DEFAULT_SOURCE,
    CompilerValidationError,
    compile_clean,
    run_regression_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("clean_reconstruction","regression_audit"),default="clean_reconstruction")
    parser.add_argument("--sam-dir",type=Path,default=DEFAULT_SAM_DIR)
    parser.add_argument("--source-image",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--moge-dir",type=Path,default=DEFAULT_MOGE)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--scene-id",default="office_test_unified_v3_clean")
    parser.add_argument("--handoff-dir",type=Path)
    parser.add_argument("--clean-output-dir",type=Path,default=DEFAULT_OUTPUT,help="clean plan used by regression_audit")
    parser.add_argument("--debug",action="store_true")
    return parser


def main(argv:list[str]|None=None) -> int:
    parser=build_parser();args=parser.parse_args(argv)
    try:
        output_dir=args.output_dir.expanduser().resolve()
        if args.mode=="clean_reconstruction":
            kwargs={
                "sam_dir":args.sam_dir,
                "source_path":args.source_image,
                "moge_dir":args.moge_dir,
                "output_dir":output_dir,
                "scene_id":args.scene_id,
            }
            if args.handoff_dir is not None:kwargs["handoff_dir"]=args.handoff_dir
            result=compile_clean(**kwargs)
            compilation=json.loads((output_dir/"compilation_report.json").read_text(encoding="utf-8"))
            summary={"success":True,"mode":args.mode,"scene_id":result["scene_id"],"object_count":result["semantic_object_count"],"output_dir":str(output_dir),"unified_scene_plan":str(output_dir/"unified_scene_plan.json"),"compilation_passed":compilation["passed"]}
        else:
            result=run_regression_audit(args.clean_output_dir.expanduser().resolve(),output_dir)
            summary={"success":True,"mode":args.mode,"output_dir":str(output_dir),"counts":result["counts"]}
        print(json.dumps(summary,separators=(",",":")))
        return 0
    except Exception as exc:
        if args.debug:traceback.print_exc()
        else:print(f"{type(exc).__name__}: {exc}",file=sys.stderr)
        return 2 if isinstance(exc,(CompilerValidationError,FileExistsError)) else 1


if __name__=="__main__":
    raise SystemExit(main())
