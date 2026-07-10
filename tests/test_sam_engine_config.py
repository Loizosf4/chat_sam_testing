from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from backend import sam_engine


def _clear_sam_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SAM_CHECKPOINT", "SAM_CHECKPOINT_PATH", "SAM_MODEL_TYPE", "SAM_DEVICE"):
        monkeypatch.delenv(name, raising=False)


def test_validate_model_config_accepts_vit_b_cpu_from_environment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "sam_vit_b_01ec64.pth"
    checkpoint.write_bytes(b"checkpoint placeholder")
    _clear_sam_env(monkeypatch)
    monkeypatch.setenv("SAM_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SAM_MODEL_TYPE", "vit_b")
    monkeypatch.setenv("SAM_DEVICE", "cpu")

    config = sam_engine.validate_model_config()

    assert config["checkpoint_path"] == str(checkpoint)
    assert config["model_type"] == "vit_b"
    assert config["device"] == "cpu"
    assert config["python_executable"] == sys.executable


def test_validate_model_config_keeps_legacy_checkpoint_path_name(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "sam_vit_h.pth"
    checkpoint.write_bytes(b"checkpoint placeholder")
    _clear_sam_env(monkeypatch)
    monkeypatch.setenv("SAM_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("SAM_DEVICE", "cpu")

    config = sam_engine.validate_model_config()

    assert config["checkpoint_path"] == str(checkpoint)


def test_validate_model_config_reports_wrong_model_type(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "sam_vit_b_01ec64.pth"
    checkpoint.write_bytes(b"checkpoint placeholder")
    _clear_sam_env(monkeypatch)
    monkeypatch.setenv("SAM_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SAM_MODEL_TYPE", "sam2")
    monkeypatch.setenv("SAM_DEVICE", "cpu")

    with pytest.raises(sam_engine.SamEngineError, match="SAM_MODEL_TYPE"):
        sam_engine.validate_model_config()


def test_validate_model_config_rejects_unavailable_cuda(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "sam_vit_b_01ec64.pth"
    checkpoint.write_bytes(b"checkpoint placeholder")
    _clear_sam_env(monkeypatch)
    monkeypatch.setenv("SAM_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SAM_MODEL_TYPE", "vit_b")
    monkeypatch.setenv("SAM_DEVICE", "cuda")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(sam_engine.SamEngineError, match="CUDA is not available"):
        sam_engine.validate_model_config()
