"""FastAPI layer (M3).

Wraps the same ``ingest_video`` / ``search`` the CLI uses -- the core modules do
not know whether a CLI or an HTTP request is driving them.

Two concurrency rules hold this together:

* **Ingestion runs in the background.** Embedding a video takes tens of
  seconds; an HTTP request cannot. ``POST /videos`` records the job as
  ``queued``, returns 202, and the work happens afterwards. This is what makes
  the ``queued`` status real rather than decorative.

* **Nothing CPU-bound touches the event loop.** FastAPI runs ``async def``
  handlers on the single event-loop thread and plain ``def`` handlers in a
  worker thread. CLIP inference never yields, so every handler and task that
  can reach it is deliberately declared ``def``. Making them ``async def``
  would stall the whole server for the duration of an ingest.

* **Recovery runs before the first request.** A restart is how a crash gets
  noticed: the lifespan hook sweeps videos the previous process abandoned and
  reconciles the index against the database, so the server never comes up
  serving a store it will only reject at query time.

SQLite connections cannot cross threads, so each request and task opens its own.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from . import config, db, recovery
from .db import Database
from .index import VectorIndex
from .ingest import content_hash, ingest_video
from .search import search as run_search

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

StatusFilter = Literal["queued", "processing", "done", "failed"]


# ---- shared state ---------------------------------------------------------


@dataclass
class AppState:
    """Long-lived objects shared across requests.

    The embedder and index are expensive to build, so they are created once at
    startup and reused. The database is *not* held here: connections are
    per-thread, so each request opens its own.
    """

    embedder: object
    index: VectorIndex
    db_path: Path
    index_dir: Path
    thumbnail_dir: Path | None = None
    upload_dir: Path | None = None
    web_dir: Path | None = None

    def open_db(self) -> Database:
        return Database(self.db_path)

    @property
    def uploads(self) -> Path:
        return Path(self.upload_dir) if self.upload_dir else config.VIDEO_DIR


def build_default_state() -> AppState:
    """Production state: the real CLIP checkpoint and the on-disk index."""
    from .embedder import get_embedder

    config.ensure_dirs()
    embedder = get_embedder()
    index = VectorIndex.load_or_create(dim=embedder.dim, directory=config.INDEX_DIR)
    return AppState(
        embedder=embedder,
        index=index,
        db_path=config.DB_PATH,
        index_dir=config.INDEX_DIR,
        web_dir=config.WEB_DIST_DIR,
    )


# ---- request / response models -------------------------------------------


class IngestRequest(BaseModel):
    path: str = Field(min_length=1, description="Path to a video file on the server")

    @field_validator("path")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must not be blank")
        return value.strip()


class VideoAccepted(BaseModel):
    video_id: str
    filename: str
    status: str


class VideoSummary(BaseModel):
    id: str
    filename: str
    status: str
    duration_sec: float
    frame_count: int
    error: str | None = None
    ingested_at: str | None = None


class VideoList(BaseModel):
    videos: list[VideoSummary]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, gt=0, le=200)
    collapse_window_sec: float | None = Field(default=None, gt=0)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class SearchHitModel(BaseModel):
    score: float
    video_id: str
    filename: str
    timestamp_sec: float
    thumbnail_url: str
    reason: str
    frame_id: int


class SearchResponse(BaseModel):
    query: str
    took_ms: float
    count: int
    results: list[SearchHitModel]


# ---- application ----------------------------------------------------------


def create_app(state: AppState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Repair the store before accepting traffic (M4).

        Safe on the event loop: nothing is being served yet, this touches no
        CLIP, and it is bounded by the index size rather than by any video.
        """
        with state.open_db() as database:
            report = recovery.recover(state.index, database, index_dir=state.index_dir)
        if not report.clean:
            logger.warning("startup recovery: %s", report.describe())
        yield

    app = FastAPI(
        title="Semantic Video Search",
        description="Search video by what is happening in the frame.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # The M5 UI will be served from a different origin in development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _run_ingest(path: Path, video_id: str) -> None:
        """Background worker. Plain ``def`` so FastAPI runs it off the event loop."""
        with state.open_db() as database:
            try:
                # index_dir makes the ingest persist its own vectors before
                # committing rows: an unsaved index is lost on restart even
                # though the database already says the video is done, and the
                # ordering is what keeps a crash here repairable.
                ingest_video(
                    path,
                    state.index,
                    database,
                    state.embedder,
                    index_dir=state.index_dir,
                    thumbnail_dir=state.thumbnail_dir,
                )
            except Exception as exc:  # noqa: BLE001
                # ingest_video already recorded `failed` with the message; the
                # job is finished either way, so do not re-raise into the task.
                logger.warning("ingest failed for %s: %s", path.name, exc)

    def _enqueue(path: Path, background: BackgroundTasks) -> VideoAccepted:
        video_id = content_hash(path)
        with state.open_db() as database:
            existing = database.get_video(video_id)
            if existing and existing.status == db.DONE:
                # FR5: already ingested. Do not re-queue, and do not overwrite
                # the done status -- that would strand duplicate vectors.
                return VideoAccepted(
                    video_id=video_id, filename=path.name, status=db.DONE
                )
            database.upsert_video(
                video_id, path.name, str(path.resolve()), 0.0, status=db.QUEUED
            )
        background.add_task(_run_ingest, path, video_id)
        return VideoAccepted(video_id=video_id, filename=path.name, status=db.QUEUED)

    # ---- ingestion ------------------------------------------------------

    @app.post("/videos", status_code=202, response_model=VideoAccepted)
    def ingest_by_path(
        request: IngestRequest, background: BackgroundTasks
    ) -> VideoAccepted:
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(404, f"no such file: {request.path}")
        if not path.is_file():
            raise HTTPException(400, f"not a file: {request.path}")
        return _enqueue(path, background)

    @app.post("/videos/upload", status_code=202, response_model=VideoAccepted)
    def ingest_upload(
        background: BackgroundTasks, file: UploadFile = File(...)
    ) -> VideoAccepted:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(
                400, f"unsupported file type {suffix!r}; expected one of {sorted(VIDEO_SUFFIXES)}"
            )

        uploads = state.uploads
        uploads.mkdir(parents=True, exist_ok=True)

        # Write to a temporary name first, then rename to the content hash, so
        # two uploads sharing a filename cannot overwrite each other and the
        # same content never lands twice under different names.
        with tempfile.NamedTemporaryFile(
            dir=uploads, suffix=".part", delete=False
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            staged = Path(tmp.name)

        destination = uploads / f"{content_hash(staged)}{suffix}"
        staged.replace(destination)
        return _enqueue(destination, background)

    # ---- status ---------------------------------------------------------

    @app.get("/videos", response_model=VideoList)
    def list_videos(status: StatusFilter | None = Query(default=None)) -> VideoList:
        with state.open_db() as database:
            rows = database.list_videos(status=status)
            return VideoList(
                videos=[
                    VideoSummary(
                        id=row.id,
                        filename=row.filename,
                        status=row.status,
                        duration_sec=row.duration_sec,
                        frame_count=database.frame_count(video_id=row.id),
                        error=row.error,
                        ingested_at=row.ingested_at,
                    )
                    for row in rows
                ]
            )

    @app.get("/videos/{video_id}/status", response_model=VideoSummary)
    def video_status(video_id: str) -> VideoSummary:
        with state.open_db() as database:
            row = database.get_video(video_id)
            if row is None:
                raise HTTPException(404, f"unknown video: {video_id}")
            return VideoSummary(
                id=row.id,
                filename=row.filename,
                status=row.status,
                duration_sec=row.duration_sec,
                frame_count=database.frame_count(video_id=row.id),
                error=row.error,
                ingested_at=row.ingested_at,
            )

    # ---- search ---------------------------------------------------------

    @app.post("/search", response_model=SearchResponse)
    def search_endpoint(request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        with state.open_db() as database:
            hits = run_search(
                request.query,
                state.index,
                database,
                state.embedder,
                top_k=request.top_k,
                collapse_window_sec=request.collapse_window_sec,
            )
        took_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(
            query=request.query,
            took_ms=round(took_ms, 2),
            count=len(hits),
            results=[
                SearchHitModel(
                    score=hit.score,
                    video_id=hit.video_id,
                    filename=hit.filename,
                    timestamp_sec=hit.timestamp_sec,
                    # A URL the browser can fetch. Never the server's filesystem
                    # path, which leaks layout and is useless to a client.
                    thumbnail_url=f"/thumbnails/{hit.frame_id}",
                    reason=hit.reason,
                    frame_id=hit.frame_id,
                )
                for hit in hits
            ],
        )

    @app.get("/thumbnails/{frame_id}")
    def thumbnail(frame_id: int) -> FileResponse:
        with state.open_db() as database:
            row = database.get_frame(frame_id)
        if row is None:
            raise HTTPException(404, f"unknown frame: {frame_id}")
        path = Path(row.thumbnail_path)
        if not path.exists():
            raise HTTPException(404, f"thumbnail missing on disk for frame {frame_id}")
        return FileResponse(path, media_type="image/jpeg")

    # ---- health ---------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        with state.open_db() as database:
            return {
                "status": "ok",
                "videos": len(database.list_videos()),
                "frames": database.frame_count(),
                "vectors": len(state.index),
                "device": getattr(state.embedder, "device", "unknown"),
            }

    # ---- the built UI ---------------------------------------------------
    #
    # Registered last, and only here. A mount at "/" matches every path, and
    # Starlette resolves routes in registration order -- so anything declared
    # after it would be unreachable, and anything before it (every route above,
    # plus /docs) still wins. Moving this block up would leave a page that
    # loads and whose every request 404s.
    ui = Path(state.web_dir) if state.web_dir else None
    if ui is not None and (ui / "index.html").is_file():
        app.mount("/", StaticFiles(directory=ui, html=True), name="ui")
    else:

        @app.get("/", include_in_schema=False)
        def ui_not_built() -> None:
            # The UI is optional -- a headless deployment is legitimate -- so
            # this is a 404 rather than a startup failure, but it says what to
            # run rather than leaving a bare Not Found.
            raise HTTPException(
                404,
                "UI not built. Run `npm --prefix web install && npm --prefix web run "
                "build`, or run the Vite dev server with `npm --prefix web run dev`.",
            )

    return app


def create_default_app() -> FastAPI:
    """Entry point for ``uvicorn sv_engine.api:create_default_app --factory``."""
    return create_app(build_default_state())
