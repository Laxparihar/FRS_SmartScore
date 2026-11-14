"""FRS SmartScore - upload two images, get a face-match score."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .face_engine import FaceEngine, FaceNotFoundError, decode_image

MAX_UPLOAD_MB = 10
STATIC_DIR = Path(__file__).parent / "static"

engine: FaceEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = FaceEngine()  # loads (and on first run downloads) the models
    yield


app = FastAPI(title="FRS SmartScore", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None}


async def _read(upload: UploadFile, label: str):
    data = await upload.read()
    if not data:
        raise HTTPException(400, f"{label}: file is empty")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{label}: file larger than {MAX_UPLOAD_MB} MB")
    try:
        return decode_image(data)
    except ValueError as e:
        raise HTTPException(400, f"{label}: {e}. Use JPG, PNG, WEBP or BMP.")


@app.post("/api/verify")
async def verify(image1: UploadFile = File(..., description="Photo / selfie"),
                 image2: UploadFile = File(..., description="ID card (Aadhaar / PAN / Passport) or second photo")):
    t0 = time.perf_counter()
    img1 = await _read(image1, "Image 1")
    img2 = await _read(image2, "Image 2")

    results = []
    for label, img in (("Image 1", img1), ("Image 2", img2)):
        try:
            results.append(await run_in_threadpool(engine.analyze, img))
        except FaceNotFoundError:
            raise HTTPException(422, f"{label}: no face detected. Upload a clear, front-facing photo.")

    a, b = results
    return {
        **engine.compare(a, b),
        "image1": a.public(),
        "image2": b.public(),
        "processing_ms": int((time.perf_counter() - t0) * 1000),
    }
