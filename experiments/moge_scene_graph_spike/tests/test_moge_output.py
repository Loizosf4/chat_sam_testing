from pathlib import Path

import pytest

from src.smoke_test_output import validate_output


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs" / "office_test" / "moge"


@pytest.mark.skipif(not (OUTPUT_DIR / "geometry.npz").exists(), reason="run MoGe inference first")
def test_persisted_moge_output() -> None:
    assert validate_output(OUTPUT_DIR) == []
