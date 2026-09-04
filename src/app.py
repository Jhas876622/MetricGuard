"""
MetricGuard - FastAPI Web Application
======================================
Endpoints:
  GET  /          → interactive dashboard HTML
  GET  /results   → latest pipeline payload as JSON
  GET  /health    → liveness check  (always 200 when process is alive)
  GET  /ready     → readiness check (503 until results.json exists)
  POST /run       → trigger a fresh pipeline run (background thread)

Run locally:
    uvicorn src.app:app --reload --port 8000

Deploy to Render:
    Start command: uvicorn src.app:app --host 0.0.0.0 --port $PORT
"""

import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SRC_DIR  = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR     = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_PATH   = OUTPUT_DIR / "results.json"
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
# Thread-safe pipeline state
# threading.Event is mutable — no `global` needed to call .set()/.clear()
# ---------------------------------------------------------------------------
_pipeline_running   = threading.Event()   # set = running, clear = idle
_pipeline_started   : float | None = None  # epoch time when current run began
_last_run_time      : float | None = None  # epoch time of last completed run
_last_run_error     : str   | None = None  # error message if last run failed


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def _run_pipeline_background(use_llm: bool) -> None:
    global _pipeline_started, _last_run_time, _last_run_error
    try:
        logger.info("Background pipeline started (use_llm=%s)", use_llm)
        from report import main as run_report
        run_report(use_llm=use_llm)
        _last_run_time  = time.time()
        _last_run_error = None
        logger.info("Background pipeline completed successfully")
    except Exception as exc:
        _last_run_error = str(exc)
        logger.error("Background pipeline failed: %s", exc)
    finally:
        _pipeline_running.clear()
        _pipeline_started = None


def _start_pipeline(use_llm: bool) -> None:
    """Set state and spawn the background thread."""
    global _pipeline_started
    _pipeline_started = time.time()
    _pipeline_running.set()
    threading.Thread(
        target=_run_pipeline_background,
        args=(use_llm,),
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Lifespan — auto-run pipeline on startup if no results exist
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _last_run_time
    if not RESULTS_PATH.exists():
        logger.info("No existing results — running initial pipeline on startup")
        _start_pipeline(use_llm=bool(os.environ.get("ANTHROPIC_API_KEY")))
    else:
        logger.info("Existing results found — skipping startup run")
        _last_run_time = RESULTS_PATH.stat().st_mtime
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
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
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness — always 200 while the process is alive."""
    return {
        "status": "ok",
        "pipeline_running": _pipeline_running.is_set(),
        "results_exist":    RESULTS_PATH.exists(),
        "last_run_time":    _last_run_time,
    }


@app.get("/ready")
def ready():
    """
    Readiness — 200 only when results are available.
    Returns 503 while the pipeline is still running on first boot.
    Used by load balancers / uptime monitors.
    """
    if not RESULTS_PATH.exists():
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": "Pipeline has not completed yet."},
        )
    return {"ready": True}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the interactive dashboard."""
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return HTMLResponse(content=DASHBOARD_PATH.read_text(encoding="utf-8"))


@app.get("/results")
def get_results():
    """
    Latest pipeline payload as JSON.
    Returns a loading state (not 404) while the first run is in progress
    so the dashboard can show a progress indicator immediately.
    """
    elapsed = (
        round(time.time() - _pipeline_started)
        if _pipeline_started else None
    )

    if _pipeline_running.is_set() and not RESULTS_PATH.exists():
        return JSONResponse(content={
            "loading":  True,
            "message":  "Pipeline is running…",
            "elapsed":  elapsed,
            "kpis":     {},
            "conflicts": [],
        })

    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No results yet — POST /run to trigger the pipeline.",
        )

    with open(RESULTS_PATH) as f:
        payload = json.load(f)

    payload["meta"] = {
        "pipeline_running": _pipeline_running.is_set(),
        "elapsed":          elapsed,
        "last_run_time":    _last_run_time,
        "last_run_error":   _last_run_error,
    }
    return JSONResponse(content=payload)


@app.post("/run")
def run_pipeline(background_tasks: BackgroundTasks, use_llm: bool = True):
    """
    Trigger a fresh pipeline run in a background thread.
    Returns 202 immediately; poll GET /results for completion.
    """
    if _pipeline_running.is_set():
        elapsed = round(time.time() - _pipeline_started) if _pipeline_started else "?"
        return JSONResponse(
            status_code=202,
            content={
                "message": "Pipeline already running.",
                "elapsed": elapsed,
            },
        )

    background_tasks.add_task(_start_pipeline, use_llm)
    logger.info("Pipeline run triggered via POST /run (use_llm=%s)", use_llm)
    return JSONResponse(
        status_code=202,
        content={
            "message":  "Pipeline started. Poll GET /results for fresh data.",
            "use_llm":  use_llm,
        },
    )
