from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
IMAGE_DIR = ROOT_DIR / "data" / "images"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/style.css")
def stylesheet() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "style.css")


@app.get("/app.js")
def javascript() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js")
