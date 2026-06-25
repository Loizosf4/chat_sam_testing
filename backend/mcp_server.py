from pathlib import Path
from shutil import copyfile
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from PIL import Image

from backend import mask_ops, sam_engine


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "images"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

mcp = FastMCP("Local SAM Mask Editor")


@mcp.tool()
def sam_health() -> dict:
    """Return local SAM server status without loading the model."""
    return sam_engine.get_status()


@mcp.tool()
def sam_load_model(
    checkpoint_path: str | None = None,
    model_type: str | None = None,
    device: str | None = None,
) -> dict:
    """Load the configured local SAM model."""
    try:
        return sam_engine.load_model(
            checkpoint_path=_empty_to_none(checkpoint_path),
            model_type=_empty_to_none(model_type),
            device=_empty_to_none(device),
        )
    except sam_engine.SamEngineError as exc:
        raise ValueError(str(exc)) from exc


@mcp.tool()
def sam_register_image(image_path: str) -> dict:
    """Register an existing local image file for later SAM prediction."""
    source_path = Path(image_path).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()

    if not source_path.is_file():
        raise ValueError(f"Image path does not point to a valid local file: {source_path}")

    extension = source_path.suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type '{extension}'.")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image_id = uuid4().hex
    stored_filename = f"{image_id}{extension}"
    stored_path = IMAGE_DIR / stored_filename

    try:
        copyfile(source_path, stored_path)
        with Image.open(stored_path) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise ValueError(f"Failed to register image: {exc}") from exc

    mask_ops.register_image(
        image_id=image_id,
        original_filename=source_path.name,
        stored_filename=stored_filename,
        width=width,
        height=height,
    )

    return {
        "image_id": image_id,
        "filename": source_path.name,
        "width": width,
        "height": height,
        "path": str(stored_path),
    }


@mcp.tool()
def sam_set_image(image_id: str) -> dict:
    """Set a registered image as the active image in the SAM predictor."""
    try:
        return sam_engine.set_image(image_id)
    except sam_engine.SamEngineError as exc:
        if "Call /load_model first" in str(exc):
            raise ValueError("SAM is not loaded. Call sam_load_model first.") from exc
        raise ValueError(str(exc)) from exc


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


if __name__ == "__main__":
    mcp.run("stdio")
