"""pallet-dimensioner — FastAPI веб-сервис: точки измерения, рабочее место оператора, документация."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import qr
from src.models import CameraSideBinding, CameraTopBinding, Post, PostCreateRequest, PostState, PostUpdateRequest
from src.store import PostStore, new_post_id, now_utc
from src.worker import PostWorkerRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    registry.stop_all()


app = FastAPI(
    title="pallet-dimensioner",
    description="Измерение габаритов паллет двумя веб-камерами + весы — точки, рабочее место, REST API",
    version="2.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

store = PostStore(BASE_DIR / "data" / "posts.json")
registry = PostWorkerRegistry()

MJPEG_BOUNDARY = b"frame"
STREAM_POLL_SECONDS = 0.08


# ---------------------------------------------------------------------------
# Страницы
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def docs_page(request: Request):
    return templates.TemplateResponse(request, "docs.html")


@app.get("/posts", include_in_schema=False)
async def posts_page(request: Request):
    posts = store.list()
    statuses = {p.id: (registry.get(p.id).get_state() if registry.get(p.id) else None) for p in posts}
    return templates.TemplateResponse(request, "posts.html", {"posts": posts, "statuses": statuses})


@app.get("/posts/{post_id}/workplace", include_in_schema=False)
async def workplace_page(request: Request, post_id: str):
    post = store.get(post_id)
    if post is None:
        raise HTTPException(404, "Точка не найдена")
    return templates.TemplateResponse(request, "workplace.html", {"post": post})


# ---------------------------------------------------------------------------
# REST API — точки
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/posts", response_model=Post)
async def create_post(req: PostCreateRequest):
    post = Post(
        id=new_post_id(req.name, store),
        name=req.name,
        camera_top=CameraTopBinding(device_id=req.camera_top_device_id),
        camera_side=CameraSideBinding(device_id=req.camera_side_device_id),
        scale_api_url=req.scale_api_url,
        wms_endpoint=req.wms_endpoint,
        created_at=now_utc(),
    )
    return store.create(post)


@app.get("/api/posts", response_model=list[Post])
async def list_posts():
    return store.list()


@app.get("/api/posts/{post_id}", response_model=Post)
async def get_post(post_id: str):
    post = store.get(post_id)
    if post is None:
        raise HTTPException(404, "Точка не найдена")
    return post


@app.patch("/api/posts/{post_id}", response_model=Post)
async def update_post(post_id: str, req: PostUpdateRequest):
    post = store.get(post_id)
    if post is None:
        raise HTTPException(404, "Точка не найдена")
    updated = post.model_copy(update=req.model_dump(exclude_unset=True))
    store.update(post_id, updated)
    registry.remove(post_id)  # следующий /start поднимет воркер уже с новым конфигом
    return updated


@app.delete("/api/posts/{post_id}", status_code=204)
async def delete_post(post_id: str):
    registry.remove(post_id)
    store.delete(post_id)
    return Response(status_code=204)


@app.post("/api/posts/{post_id}/start", response_model=Post)
async def start_post(post_id: str):
    post = store.get(post_id)
    if post is None:
        raise HTTPException(404, "Точка не найдена")
    registry.get_or_create(post).start()
    return post


@app.post("/api/posts/{post_id}/stop", status_code=204)
async def stop_post(post_id: str):
    worker = registry.get(post_id)
    if worker:
        worker.stop()
    return Response(status_code=204)


@app.get("/api/posts/{post_id}/state")
async def get_post_state(post_id: str):
    worker = registry.get(post_id)
    if worker is None:
        raise HTTPException(404, "Точка не запущена — вызови /api/posts/{id}/start")
    return worker.get_state()


# ---------------------------------------------------------------------------
# Видео и QR
# ---------------------------------------------------------------------------

def _mjpeg_stream(worker, camera: str):
    while True:
        frame = worker.get_frame_jpeg(camera)
        if frame is not None:
            yield (
                b"--" + MJPEG_BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(STREAM_POLL_SECONDS)


@app.get("/posts/{post_id}/stream/{camera}")
async def stream_camera(post_id: str, camera: str):
    if camera not in ("top", "side"):
        raise HTTPException(404, "Камера должна быть top или side")
    worker = registry.get(post_id)
    if worker is None or not worker.is_running():
        raise HTTPException(409, "Точка не запущена")
    return StreamingResponse(
        _mjpeg_stream(worker, camera),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY.decode()}",
    )


@app.get("/posts/{post_id}/qr.png")
async def post_qr(post_id: str):
    post = store.get(post_id)
    if post is None:
        raise HTTPException(404, "Точка не найдена")
    worker = registry.get(post_id)
    state = worker.get_state() if worker else PostState(post_id=post_id)
    png = qr.generate_png(post, state)
    return Response(content=png, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8012, log_level="info")
