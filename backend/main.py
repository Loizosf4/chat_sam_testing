from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from backend import sam_engine


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
IMAGE_DIR = ROOT_DIR / "data" / "images"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SetImageRequest(BaseModel):
    image_id: str


class PredictRequest(BaseModel):
    image_id: str = ""
    points: list[list[float]] = Field(default_factory=list)
    point_labels: list[int] = Field(default_factory=list)
    box: list[float] | None = None
    multimask_output: bool = True


app = FastAPI(title="Local SAM Mask Editor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return health()


@app.post("/upload_image")
async def upload_image(request: Request, image: UploadFile = File(...)) -> dict[str, str | int]:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    extension = Path(image.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image_id = uuid4().hex
    stored_filename = f"{image_id}{extension}"
    image_path = IMAGE_DIR / stored_filename

    content = await image.read()
    image_path.write_bytes(content)

    try:
        with Image.open(image_path) as uploaded:
            width, height = uploaded.size
            uploaded.verify()
    except Exception as exc:
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    image_url = str(request.url_for("images", path=stored_filename))

    return {
        "image_id": image_id,
        "filename": image.filename,
        "width": width,
        "height": height,
        "url": image_url,
    }


@app.post("/load_model")
def load_model() -> dict[str, str | bool]:
    try:
        return sam_engine.load_model()
    except sam_engine.SamEngineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/set_image")
def set_image(payload: SetImageRequest) -> dict[str, str | int | bool]:
    try:
        return sam_engine.set_image(payload.image_id)
    except sam_engine.SamEngineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    try:
        return sam_engine.predict(
            image_id=payload.image_id,
            points=payload.points,
            point_labels=payload.point_labels,
            box=payload.box,
            multimask_output=payload.multimask_output,
        )
    except sam_engine.SamEngineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/mask/{mask_id}.png")
def get_mask(mask_id: str) -> FileResponse:
    try:
        return FileResponse(sam_engine.get_mask_path(mask_id), media_type="image/png")
    except sam_engine.SamEngineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/style.css")
def stylesheet() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "style.css")


@app.get("/app.js")
def javascript() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js")
