"""The five specialised agents. Each has one job, a restricted toolset and a typed hand-off."""
from __future__ import annotations
import itertools
import re
from concurrent.futures import ThreadPoolExecutor
from . import tools
from .models import Claim, Evidence, Section, Source, SubQuestion
from .resilience import retry
from . import verifier
from .textutil import bm25_rank, jaccard, normalize, sentences, tokens

FACETS = {
    "overview": ("Overview", "definition refers is"),
    "how": ("How it works", "works process method mechanism approach"),
    "challenges": ("Challenges & limitations", "challenges limitations problems criticism risks"),
    "applications": ("Applications", "applications used uses examples"),
    "recent": ("Recent developments", "recent advances development progress 2024 2025"),
    "history": ("History", "history origin invented developed early"),
    "compare": ("Alternatives & comparison", "compared alternative versus difference"),
}
DEPTH = {  # facets, max sources, evidence per section
    "quick": (["overview", "how", "challenges"], 5, 2),
    "standard": (["overview", "how", "applications", "challenges", "recent"], 8, 3),
    "deep": (["overview", "history", "how", "applications", "challenges", "compare", "recent"], 12, 3),
}


class Ctx:
    """Everything an agent needs: LLM, blackboard, memory, event emitter."""
    def __init__(self, llm, board, emit, memory=None, session_id=None, depth="standard"):
        self.llm, self.board, self.emit = llm, board, emit
        self.memory, self.session_id, self.depth = memory, session_id, depth
        self._sid, self._eid = itertools.count(1), itertools.count(1)

    def retry(self, node, label, fn, attempts=3):
        def on_retry(i, e, d):
            self.emit("retry", node, message=f"{label}: {type(e).__name__} — retry {i} in {d:.1f}s")
        return retry(fn, attempts=attempts, on_retry=on_retry)


# ───────────────────────── 1. Planner ─────────────────────────
def planner(ctx: Ctx) -> list[SubQuestion]:
    q = ctx.board.question
    facets, _, _ = DEPTH[ctx.depth]
    topic = " ".join(tokens(q)) or q
    ctx.board.topic = topic
    plan = None
    if ctx.llm.available:
        try:
            prior = [t["question"] for t in ctx.memory.turns(ctx.session_id)] if ctx.memory and ctx.session_id else []
            data = ctx.retry("planner", "LLM plan", lambda: ctx.llm.json(
                "You are a research planner. Decompose the question into focused sub-questions that together cover it.",
                f'Question: {q}\nPrevious questions in this session: {prior}\n'
                f'Return {{"topic":"<2-5 word search topic>","subquestions":[{{"heading":"...","query":"<web search query>"}}]}} '
                f'with exactly {len(facets)} subquestions.', 800), attempts=2)
            subs = data["subquestions"][: len(facets)]
            ctx.board.topic = data.get("topic") or topic
            plan = [SubQuestion(f"Q{i+1}", s["heading"][:60], s["query"]) for i, s in enumerate(subs)]
        except Exception as e:  # noqa: BLE001
            ctx.emit("warn", "planner", message=f"LLM planning failed ({type(e).__name__}); using heuristic plan")
    if not plan:
        plan = [SubQuestion(f"Q{i+1}", FACETS[f][0], f"{topic} {FACETS[f][1].split()[0]}") for i, f in enumerate(facets)]
        plan = [SubQuestion(p.id, p.heading, f"{topic} {FACETS[f][1]}") for p, f in zip(plan, facets)]
    ctx.board.subqs = plan
    return plan


# ───────────────────────── 2. Searcher ─────────────────────────
def searcher(ctx: Ctx, sq: SubQuestion, per_query: int = 4) -> list[Source]:
    found: list[dict] = []
    topic = ctx.board.topic

    def wiki():
        return tools.call("searcher", "wikipedia_search", topic if sq.id == "Q1" else sq.query, per_query)

    def web():
        return tools.call("searcher", "web_search", sq.query, per_query)

    def arxiv():
        return tools.call("searcher", "arxiv_search", ctx.board.topic if sq.id == "Q1" else sq.query, 3)

    providers = [("wikipedia", wiki), ("web", web)]
    if sq.id in ("Q1", ctx.board.subqs[-1].id):  # arXiv is rate-limited (1 req/3s): use for 2 sub-questions
        providers.append(("arxiv", arxiv))
    for name, fn in providers:
        try:
            found += ctx.retry("searcher", f"{sq.id} {name}", fn, attempts=2)
        except Exception as e:  # noqa: BLE001
            ctx.emit("warn", "searcher", message=f"{sq.id}: {name} search unavailable ({type(e).__name__})")
    out = []
    for r in found:
        s = Source(id=f"S{next(ctx._sid)}", url=r["url"], title=r["title"], snippet=r["snippet"],
                   origin=r["origin"], subq=sq.id)
        if ctx.board.add_source(s):
            out.append(s)
    ctx.emit("log", "searcher", message=f"{sq.id} “{sq.heading}”: {len(out)} new sources")
    return out


def pick_sources(ctx: Ctx) -> list[Source]:
    """Round-robin across sub-questions so every facet is represented; prefer diverse origins."""
    _, cap, _ = DEPTH[ctx.depth]
    by_q: dict[str, list[Source]] = {}
    for s in ctx.board.sources.values():
        by_q.setdefault(s.subq, []).append(s)
    chosen, rr = [], 0
    while len(chosen) < cap and any(by_q.values()):
        for q in list(by_q):
            if by_q[q] and len(chosen) < cap:
                chosen.append(by_q[q].pop(0))
        rr += 1
    return chosen


# ───────────────────────── 3. Extractor ─────────────────────────
def extractor(ctx: Ctx, src: Source) -> list[Evidence]:
    def get():
        if src.origin == "arxiv":
            return src.snippet  # abstract returned by the search tool
        if src.origin == "wikipedia":
            return tools.call("extractor", "wikipedia_text", src.title)
        return tools.call("extractor", "fetch_page", src.url)

    try:
        src.text = ctx.retry("extractor", f"fetch {src.id}", get, attempts=2)
        src.status = "fetched"
    except Exception as e:  # noqa: BLE001
        src.status = "failed"
        ctx.emit("warn", "extractor", message=f"{src.id} {src.title[:40]}: fetch failed ({type(e).__name__}) — skipped")
        return []
    sents = sentences(src.text)
    if not sents:
        src.status = "failed"
        return []
    topic_toks = set(tokens(ctx.board.topic))
    out: list[Evidence] = []
    for sq in ctx.board.subqs:
        facet_kw = next((v[1] for v in FACETS.values() if v[0] == sq.heading), sq.query)
        ranked = bm25_rank(f"{ctx.board.topic} {ctx.board.topic} {facet_kw} {sq.query}", sents)
        cand = [(sents[i], sc) for i, sc in ranked[:12] if sc > 0 and topic_toks & set(tokens(sents[i]))]
        if ctx.llm.available and cand:
            cand = _llm_filter(ctx, sq, cand)
        for text, sc in cand[:2]:
            out.append(Evidence("", src.id, sq.id, text, round(sc, 3)))
    ctx.emit("log", "extractor", message=f"{src.id} {src.title[:48]}: {len(out)} passages")
    return out


def _llm_filter(ctx, sq, cand):
    try:
        listing = "\n".join(f"{i}: {t}" for i, (t, _) in enumerate(cand))
        d = ctx.llm.json("Select sentences that directly and factually answer the sub-question. Never rewrite them.",
                         f'Sub-question: {sq.heading} — {sq.query}\n{listing}\nReturn {{"keep":[indices]}} (max 3).', 300)
        keep = [cand[i] for i in d.get("keep", []) if isinstance(i, int) and 0 <= i < len(cand)]
        return keep or cand
    except Exception:  # noqa: BLE001 - degrade to lexical ranking
        return cand


def register_evidence(ctx: Ctx, items: list[Evidence]) -> None:
    """Deduplicate near-identical passages, then assign stable ids."""
    kept: list[Evidence] = []
    for e in sorted(ctx.board.evidence + items, key=lambda e: -e.score):
        if all(jaccard(e.quote, k.quote) < 0.7 for k in kept):
            kept.append(e)
    for e in kept:
        e.id = f"E{next(ctx._eid)}"
    ctx.board.evidence = kept


# ───────────────────────── 4. Synthesizer ─────────────────────────
def synthesizer(ctx: Ctx) -> tuple[list[Claim], list[Section], str]:
    _, _, per_sec = DEPTH[ctx.depth]
    ev = ctx.board.evidence
    mode = "extractive"
    if ctx.llm.available and ev:
        try:
            return (*_llm_synthesis(ctx), "generative")
        except Exception as e:  # noqa: BLE001
            ctx.emit("warn", "synthesizer", message=f"LLM synthesis failed ({type(e).__name__}); falling back to extractive")
    top = sorted(ev, key=lambda e: -e.score)
    summary, seen = [], set()
    for e in top:
        if e.subq not in seen and len(summary) < 3:
            seen.add(e.subq)
            summary.append(Claim(e.quote, [e.id]))
    used = {c.evidence[0] for c in summary}
    sections = []
    for sq in ctx.board.subqs:
        mine = sorted([e for e in ev if e.subq == sq.id and e.id not in used], key=lambda e: -e.score)[:per_sec]
        if mine:
            sections.append(Section(sq.heading, [Claim(e.quote, [e.id]) for e in mine]))
    return summary, sections, mode


def _llm_synthesis(ctx):
    ev = ctx.board.evidence[:60]
    listing = "\n".join(f"[{e.id}] ({e.subq}) {e.quote}" for e in ev)
    heads = [f"{s.id}: {s.heading}" for s in ctx.board.subqs]
    d = ctx.retry("synthesizer", "LLM synthesis", lambda: ctx.llm.json(
        "You write rigorous research briefs. Use ONLY the supplied evidence. Every claim must cite evidence ids it is "
        "directly supported by. Paraphrase concisely; never add facts that are not in the evidence. Skip sections with no evidence.",
        f'Question: {ctx.board.question}\nSections: {heads}\nEvidence:\n{listing}\n'
        'Return {"summary":[{"text":"","evidence":["E1"]}],"sections":[{"heading":"","claims":[{"text":"","evidence":["E1"]}]}]} '
        "with 2-4 summary claims and 2-4 claims per section.", 3500), attempts=2)
    conv = lambda c: Claim(str(c["text"]), [x for x in c.get("evidence", []) if isinstance(x, str)])
    return ([conv(c) for c in d.get("summary", [])],
            [Section(s["heading"], [conv(c) for c in s.get("claims", [])]) for s in d.get("sections", [])])


# ───────────────────────── 5. Citer / verifier ─────────────────────────
def citer(ctx: Ctx, summary: list[Claim], sections: list[Section]) -> dict:
    ev = {e.id: e for e in ctx.board.evidence}
    src = ctx.board.sources
    numbering: dict[str, int] = {}
    stats = {"claims": 0, "verified": 0, "dropped": 0, "method": "lexical"}

    def check(c: Claim) -> bool:
        stats["claims"] += 1
        cited = [ev[i] for i in c.evidence if i in ev]
        ok_ev = [e for e in cited if normalize(e.quote) in normalize(src[e.source_id].text)]  # quote really exists
        if not ok_ev:
            return False
        c.support, method = verifier.score(c.text, [e.quote for e in ok_ev])
        stats["method"] = method
        c.verified = c.support >= (0.5 if method == "learned" else 0.45)
        c.evidence = [e.id for e in ok_ev]
        return c.verified

    def render(c: Claim) -> dict:
        nums = []
        for i in c.evidence:
            n = numbering.setdefault(ev[i].source_id, len(numbering) + 1)
            if n not in nums:
                nums.append(n)
        return {"text": c.text, "cites": sorted(nums), "support": c.support,
                "quotes": [{"n": numbering[ev[i].source_id], "quote": ev[i].quote} for i in c.evidence]}

    def process(cs):
        out = []
        for c in cs:
            if check(c):
                stats["verified"] += 1
                out.append(render(c))
            else:
                stats["dropped"] += 1
                ctx.emit("warn", "citer", message=f"Dropped unsupported claim: “{c.text[:70]}…”")
        return out

    r_summary = process(summary)
    r_sections = [{"heading": s.heading, "claims": cs} for s in sections if (cs := process(s.claims))]
    refs = [{"n": n, **src[sid].public()} for sid, n in sorted(numbering.items(), key=lambda x: x[1])]
    stats["grounding"] = round(stats["verified"] / stats["claims"], 2) if stats["claims"] else 0.0
    stats["sources_cited"] = len(refs)
    return {"summary": r_summary, "sections": r_sections, "sources": refs, "stats": stats}


# ───────────────────────── 6. Critic (gap analysis → second search round) ─────────────────────────
def critic_gaps(ctx: Ctx, min_evidence: int = 2) -> list[SubQuestion]:
    """Find sub-questions the evidence doesn't cover well and propose reformulated queries."""
    gaps = []
    for sq in ctx.board.subqs:
        n = sum(1 for e in ctx.board.evidence if e.subq == sq.id)
        if n >= min_evidence:
            continue
        alt = f"{ctx.board.topic} {re.sub(r'[^A-Za-z ]', ' ', sq.heading)}"
        if ctx.llm.available:
            try:
                alt = ctx.llm.json("Rewrite a failed web search query so it finds different, more specific sources.",
                                   f'Topic: {ctx.board.topic}\nSub-question: {sq.heading}\nFailed query: {sq.query}\nReturn {{"query":"..."}}', 150)["query"]
            except Exception:  # noqa: BLE001
                pass
        gaps.append(SubQuestion(sq.id, sq.heading, alt))
    return gaps
