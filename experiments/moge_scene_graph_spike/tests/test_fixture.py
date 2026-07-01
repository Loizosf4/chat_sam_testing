from pathlib import Path

from src.validate_fixture import validate_fixture


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_office_fixture_is_valid() -> None:
    manifest = EXPERIMENT_ROOT / "inputs" / "office_test" / "manifest.json"
    assert validate_fixture(manifest) == []
