"""Orchestrator: runs the agent graph  planner → searcher×N → extractor×M → synthesizer → citer,
with parallel fan-out, retries, graceful degradation, shared memory, and a full event log."""
from __future__ import annotations
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from . import agents
from .config import settings
from .llm import get_llm
from .memory import Blackboard, SessionMemory

GRAPH = {
    "nodes": [
        {"id": "planner", "label": "Planner", "desc": "Decomposes the question"},
        {"id": "searcher", "label": "Searcher", "desc": "Wiki + arXiv + web"},
        {"id": "extractor", "label": "Extractor", "desc": "Fetches & pulls quotes"},
        {"id": "critic", "label": "Critic", "desc": "Finds coverage gaps"},
        {"id": "synthesizer", "label": "Synthesizer", "desc": "Writes the brief"},
        {"id": "citer", "label": "Citer", "desc": "Verifies every claim"},
    ],
    "edges": [["planner", "searcher"], ["searcher", "extractor"], ["extractor", "critic"], ["critic", "synthesizer"], ["synthesizer", "citer"]],
}
MEMORY = SessionMemory()


class Run:
    def __init__(self, question: str, depth: str, session_id: str | None):
        self.id = uuid.uuid4().hex[:10]
        self.question, self.depth = question, depth
        self.session_id = session_id or self.id
        self.status = "queued"
        self.events: list[dict] = []
        self.report: dict | None = None
        self.error: str | None = None
        self.t0 = time.time()
        self._lock = threading.Lock()
        os.makedirs(settings().runs_dir, exist_ok=True)
        self._path = os.path.join(settings().runs_dir, f"{self.id}.jsonl")

    def emit(self, type_: str, node: str | None = None, **data):
        ev = {"i": len(self.events), "t": round(time.time() - self.t0, 2), "type": type_, "node": node, **data}
        with self._lock:
            ev["i"] = len(self.events)
            self.events.append(ev)
            with open(self._path, "a") as f:
                f.write(json.dumps(ev) + "\n")

    def summary(self) -> dict:
        return {"id": self.id, "question": self.question, "depth": self.depth, "status": self.status,
                "session_id": self.session_id, "elapsed": round(time.time() - self.t0, 1), "error": self.error}


def _stage(run: Run, node: str, fn, **info):
    run.emit("node_start", node, **info)
    t = time.time()
    try:
        res = fn()
    except Exception as e:  # noqa: BLE001
        run.emit("node_error", node, message=f"{type(e).__name__}: {e}")
        raise
    run.emit("node_end", node, ms=int((time.time() - t) * 1000))
    return res


def execute(run: Run, llm=None, memory: SessionMemory = MEMORY) -> None:
    llm = llm or get_llm()
    run.status = "running"
    run.emit("run_start", None, llm=llm.name, depth=run.depth, graph=GRAPH)
    board = Blackboard(question=run.question)
    ctx = agents.Ctx(llm, board, run.emit, memory, run.session_id, run.depth)
    try:
        subqs = _stage(run, "planner", lambda: agents.planner(ctx))
        run.emit("plan", "planner", subqs=[s.__dict__ for s in subqs])

        recalled = memory.recall(run.session_id, run.question)
        if recalled:
            run.emit("log", "planner", message=f"Memory: recalled {len(recalled)} passages from earlier turns")

        def search_all():
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda sq: agents.searcher(ctx, sq), subqs))
            if not board.sources:
                raise RuntimeError("No sources found — check network connectivity or rephrase the question")
            chosen = agents.pick_sources(ctx)
            run.emit("sources", "searcher", sources=[s.public() for s in chosen])
            return chosen
        chosen = _stage(run, "searcher", search_all, queries=len(subqs))

        def extract_all():
            with ThreadPoolExecutor(max_workers=6) as pool:
                batches = list(pool.map(lambda s: agents.extractor(ctx, s), chosen))
            items = [e for b in batches for e in b]
            agents.register_evidence(ctx, items)
            if not board.evidence:
                raise RuntimeError("No usable evidence could be extracted from the sources")
            run.emit("sources", "extractor", sources=[s.public() for s in chosen])
            return board.evidence
        ev = _stage(run, "extractor", extract_all, sources=len(chosen))
        run.emit("metric", "extractor", evidence=len(ev), fetched=sum(s.status == "fetched" for s in chosen))

        def critique():
            gaps = agents.critic_gaps(ctx)
            if not gaps:
                run.emit("log", "critic", message="Coverage complete — no gaps")
                return 0
            run.emit("log", "critic", message=f"{len(gaps)} thin sub-question(s): " + ", ".join(g.heading for g in gaps) + " → second search round")
            before = set(board.sources)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda g: agents.searcher(ctx, g), gaps))
            fresh = [s for sid, s in board.sources.items() if sid not in before][:4]
            with ThreadPoolExecutor(max_workers=4) as pool:
                batches = list(pool.map(lambda s: agents.extractor(ctx, s), fresh))
            n0 = len(board.evidence)
            agents.register_evidence(ctx, [e for b in batches for e in b])
            run.emit("log", "critic", message=f"+{len(board.evidence) - n0} passages from {len(fresh)} new sources")
            run.emit("sources", "critic", sources=[s.public() for s in list(chosen) + fresh])
            chosen.extend(fresh)
            return len(gaps)
        _stage(run, "critic", critique)
        run.emit("metric", "critic", evidence=len(board.evidence), fetched=sum(s.status == "fetched" for s in chosen))

        summary, sections, mode = _stage(run, "synthesizer", lambda: agents.synthesizer(ctx))
        out = _stage(run, "citer", lambda: agents.citer(ctx, summary, sections))
        if out["stats"]["verified"] == 0:
            raise RuntimeError("No claim survived citation verification")

        out.update(question=run.question, mode=mode, llm=llm.name, depth=run.depth,
                   elapsed=round(time.time() - run.t0, 1))
        out["markdown"] = to_markdown(run.question, out)
        run.report = out
        memory.remember(run.session_id, run.question, " ".join(c["text"] for c in out["summary"]),
                        [{"quote": e.quote} for e in ev])
        run.status = "done"
        run.emit("report", "citer", stats=out["stats"], elapsed=out["elapsed"])
    except Exception as e:  # noqa: BLE001
        run.status, run.error = "failed", str(e)
        run.emit("run_failed", None, message=str(e))


def to_markdown(question: str, r: dict) -> str:
    cite = lambda c: "".join(f"[{n}]" for n in c["cites"])
    L = [f"# {question}", "", "## Key findings", ""]
    L += [f"- {c['text']} {cite(c)}" for c in r["summary"]]
    for s in r["sections"]:
        L += ["", f"## {s['heading']}", ""] + [f"- {c['text']} {cite(c)}" for c in s["claims"]]
    L += ["", "## References", ""] + [f"{s['n']}. [{s['title']}]({s['url']})" for s in r["sources"]]
    st = r["stats"]
    L += ["", f"> Grounding {int(st['grounding']*100)}% · {st['verified']}/{st['claims']} claims verified · "
          f"{st['sources_cited']} sources · mode: {r['mode']}"]
    return "\n".join(L)
