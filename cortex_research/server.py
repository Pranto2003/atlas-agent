import asyncio
import json
import os
import threading
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from . import __version__, verifier
from .config import settings
from .llm import get_llm
from .orchestrator import GRAPH, Run, execute

app = FastAPI(title="Cortex Research", version=__version__)
RUNS: dict[str, Run] = {}
SLOTS = threading.BoundedSemaphore(settings().max_concurrent_runs)
STATIC = os.path.join(os.path.dirname(__file__), "static")


class RunRequest(BaseModel):
    question: str = Field(min_length=8, max_length=500)
    depth: Literal["quick", "standard", "deep"] = "standard"
    session_id: str | None = Field(default=None, max_length=40)


def _worker(run: Run):
    with SLOTS:
        execute(run)


def _start(req: RunRequest) -> Run:
    run = Run(req.question.strip(), req.depth, req.session_id)
    RUNS[run.id] = run
    threading.Thread(target=_worker, args=(run,), daemon=True).start()
    return run


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "llm": get_llm().name}


@app.get("/api/model")
def model_info():
    return verifier.info()


@app.get("/api/graph")
def graph():
    return GRAPH


@app.post("/api/runs", status_code=202)
def create_run(req: RunRequest):
    return _start(req).summary()


@app.get("/api/runs")
def list_runs():
    return [r.summary() for r in sorted(RUNS.values(), key=lambda r: -r.t0)][:30]


def _get(run_id: str) -> Run:
    if run_id not in RUNS:
        raise HTTPException(404, "run not found")
    return RUNS[run_id]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    r = _get(run_id)
    return {**r.summary(), "report": r.report}


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str):
    r = _get(run_id)

    async def gen():
        i = 0
        while True:
            while i < len(r.events):
                yield f"data: {json.dumps(r.events[i])}\n\n"
                i += 1
            if r.status in ("done", "failed") and i >= len(r.events):
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(0.15)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/runs/{run_id}/report.md", response_class=PlainTextResponse)
def report_md(run_id: str):
    r = _get(run_id)
    if not r.report:
        raise HTTPException(409, "report not ready")
    return r.report["markdown"]


@app.post("/api/webhook/n8n")
async def n8n_webhook(req: RunRequest):
    """Synchronous endpoint for n8n / Zapier: blocks until the report is ready."""
    run = _start(req)
    while run.status not in ("done", "failed"):
        await asyncio.sleep(0.3)
    if run.status == "failed":
        raise HTTPException(502, run.error)
    return {"run_id": run.id, **run.report}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
