"""
MetricGuard - FastAPI Web Application
======================================
Serves the dashboard over HTTP and exposes the pipeline as an API.

Endpoints:
  GET  /          → serves the interactive dashboard HTML
  GET  /results   → returns the latest pipeline payload as JSON
  POST /run       → triggers the full pipeline and returns fresh results
  GET  /health    → liveness check (used by Render health checks)

Run locally:
    uvicorn src.app:app --reload --port 8000
    # then open http://localhost:8000

Deploy to Render:
    - Set environment variable ANTHROPIC_API_KEY in Render dashboard
    - Start command: uvicorn src.app:app --host 0.0.0.0 --port $PORT
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Path setup — make src/ importable when running from project root
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "results.json"
DASHBOARD_PATH = OUTPUT_DIR / "dashboard.html"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler (replaces deprecated @app.on_event("startup")).
    Runs the pipeline once on startup if no results exist yet.

    On Render free tier the neural model download can stall.
    Setting HF_HUB_OFFLINE=1 in the environment forces TF-IDF fallback
    so the pipeline always completes within a few seconds.
    """
    global _last_run_time
    if not RESULTS_PATH.exists():
        logger.info("No existing results — running initial pipeline on startup")
        # Run with use_llm=False on startup so the server responds immediately.
        # The user can trigger a full LLM run via POST /run once the server is up.
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
        _pipeline_running.set()
        thread = threading.Thread(
            target=_run_pipeline_background,
            args=(use_llm,),
            daemon=True,
        )
        thread.start()
    else:
        logger.info("Existing results found at %s — skipping startup run", RESULTS_PATH)
        _last_run_time = RESULTS_PATH.stat().st_mtime
    yield   # app runs here


app = FastAPI(
    title="MetricGuard",
    description="AI-powered metric consistency auditor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Thread-safe pipeline state
# ---------------------------------------------------------------------------
# threading.Event is used instead of a plain bool so reads and writes are
# atomic across threads. _pipeline_running.is_set() == True means a run is
# in progress; clear() marks it done. A plain bool is not thread-safe in
# CPython when modified from a background thread and read from the main thread.
# ---------------------------------------------------------------------------
_pipeline_running = threading.Event()   # set = running, clear = idle
_last_run_time: float | None = None
_last_run_error: str | None = None


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def _run_pipeline(use_llm: bool = True) -> dict:
    """Run the full MetricGuard pipeline and return the payload."""
    from report import main as run_report
    payload = run_report(use_llm=use_llm)
    return payload


def _run_pipeline_background(use_llm: bool) -> None:
    global _last_run_time, _last_run_error
    try:
        logger.info("Background pipeline started (use_llm=%s)", use_llm)
        _run_pipeline(use_llm=use_llm)
        _last_run_time = time.time()
        _last_run_error = None
        logger.info("Background pipeline completed successfully")
    except Exception as exc:
        _last_run_error = str(exc)
        logger.error("Background pipeline failed: %s", exc)
    finally:
        _pipeline_running.clear()   # mark idle — atomic on threading.Event


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check for Render and other platforms."""
    return {
        "status": "ok",
        "pipeline_running": _pipeline_running.is_set(),
        "last_run_time": _last_run_time,
        "results_exist": RESULTS_PATH.exists(),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the interactive dashboard."""
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return HTMLResponse(content=DASHBOARD_PATH.read_text(encoding="utf-8"))


@app.get("/results")
def get_results():
    """
    Return the latest pipeline results as JSON.
    The dashboard calls this endpoint on load and every 30 seconds
    (replaces the static data.js approach used in the local file version).
    """
    if _pipeline_running.is_set() and not RESULTS_PATH.exists():
        return JSONResponse(content={
            "loading": True,
            "message": "Pipeline is running, please wait...",
            "kpis": {},
            "conflicts": [],
        })

    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No results yet. POST /run to trigger the pipeline."
        )

    with open(RESULTS_PATH) as f:
        payload = json.load(f)

    payload["meta"] = {
        "last_run_time": _last_run_time,
        "pipeline_running": _pipeline_running.is_set(),
        "last_run_error": _last_run_error,
    }
    return JSONResponse(content=payload)


@app.post("/run")
def run_pipeline(
    background_tasks: BackgroundTasks,
    use_llm: bool = True,
):
    """
    Trigger a fresh pipeline run.
    Runs in a background thread so the response returns immediately.
    Poll GET /results or GET /health to check completion.

    Query params:
      use_llm=false   skip LLM calls (faster, works without ANTHROPIC_API_KEY)
    """
    global _pipeline_running

    if _pipeline_running.is_set():
        return JSONResponse(
            status_code=202,
            content={"message": "Pipeline already running. Poll /health for status."},
        )

    _pipeline_running.set()
    background_tasks.add_task(_run_pipeline_background, use_llm)
    logger.info("Pipeline run triggered via POST /run (use_llm=%s)", use_llm)

    return JSONResponse(
        status_code=202,
        content={
            "message": "Pipeline started. Poll GET /results for fresh data.",
            "use_llm": use_llm,
        },
    )
