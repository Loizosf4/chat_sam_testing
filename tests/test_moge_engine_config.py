from __future__ import annotations

from pathlib import Path

import pytest

from backend import moge_engine
from backend.moge_worker import WorkerError, _check_compatibility_fix


def _disable_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(moge_engine, "load_dotenv", lambda *_args, **_kwargs: None)


def _set_moge_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    python: Path,
    repository: Path,
    checkpoint: Path,
    device: str = "cuda",
    allow_cpu: str = "false",
) -> None:
    _disable_dotenv(monkeypatch)
    monkeypatch.setenv("MOGE_PYTHON", str(python))
    monkeypatch.setenv("MOGE_REPOSITORY", str(repository))
    monkeypatch.setenv("MOGE_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("MOGE_VERSION", "v2")
    monkeypatch.setenv("MOGE_DEVICE", device)
    monkeypatch.setenv("MOGE_MODEL", "moge-2-vitl-normal")
    monkeypatch.setenv("MOGE_ALLOW_CPU", allow_cpu)


def _fake_moge_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    repository = tmp_path / "moge-repository"
    repository.mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    return python, repository, checkpoint


def test_validate_model_config_requires_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_dotenv(monkeypatch)
    for name in ("MOGE_PYTHON", "MOGE_REPOSITORY", "MOGE_CHECKPOINT"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(moge_engine.MogeEngineError, match="MOGE_PYTHON"):
        moge_engine.validate_model_config()


def test_validate_model_config_rejects_implicit_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python, repository, checkpoint = _fake_moge_paths(tmp_path)
    _set_moge_env(
        monkeypatch,
        python=python,
        repository=repository,
        checkpoint=checkpoint,
        device="cpu",
        allow_cpu="false",
    )

    with pytest.raises(moge_engine.MogeEngineError, match="MOGE_ALLOW_CPU"):
        moge_engine.validate_model_config()


def test_run_inference_invokes_external_worker_and_parses_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python, repository, checkpoint = _fake_moge_paths(tmp_path)
    _set_moge_env(monkeypatch, python=python, repository=repository, checkpoint=checkpoint)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "image-1.png"
    image.write_bytes(b"not used by mocked subprocess")
    monkeypatch.setattr(moge_engine, "IMAGE_DIR", image_dir)

    calls = []

    def fake_request(config, request, timeout):
        calls.append((config, request, timeout))
        return {"success": True, "output_paths": {"geometry_npz": "geometry.npz"}}

    monkeypatch.setattr(moge_engine, "_request_service", fake_request)

    result = moge_engine.run_inference(
        "image-1",
        output_dir=str(tmp_path / "moge-output"),
        requested_outputs=["geometry_npz"],
        resolution_level=7,
    )

    config, request, timeout = calls[0]
    assert config.python == python
    assert request["command"] == "infer"
    assert request["output_type"] == ["geometry_npz"]
    assert request["resolution_level"] == 7
    assert timeout == moge_engine.WORKER_REQUEST_TIMEOUT_SECONDS
    assert result["output_paths"]["geometry_npz"] == "geometry.npz"


def test_worker_requires_windows_amd_compatibility_fix(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    utils = repository / "moge" / "model" / "utils.py"
    utils.parent.mkdir(parents=True)
    utils.write_text("from typing import *\n", encoding="utf-8")

    with pytest.raises(WorkerError, match="annotations"):
        _check_compatibility_fix(repository)

    utils.write_text("from __future__ import annotations\nfrom typing import *\n", encoding="utf-8")
    _check_compatibility_fix(repository)
