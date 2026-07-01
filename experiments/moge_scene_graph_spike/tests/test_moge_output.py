from pathlib import Path
import unittest

from src.smoke_test_output import validate_output


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs" / "office_test" / "moge"


@unittest.skipUnless((OUTPUT_DIR / "geometry.npz").exists(), "run MoGe inference first")
def test_persisted_moge_output() -> None:
    assert validate_output(OUTPUT_DIR) == []
