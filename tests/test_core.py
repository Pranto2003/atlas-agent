import pytest
from cortex_research import agents, tools
from cortex_research.llm import OfflineLLM
from cortex_research.memory import Blackboard
from cortex_research.models import Claim, Evidence, Section, Source
from cortex_research.orchestrator import Run, execute
from cortex_research.resilience import retry
from cortex_research.textutil import bm25_rank, sentences

TEXT = ("Retrieval-augmented generation combines a language model with an external document retriever to ground answers. "
        "Critics note that retrieval quality limits the final answer quality of generation systems in practice. "
        "The weather today in some city is generally mild and pleasant for walking around outside.")


def test_bm25_prefers_relevant():
    s = sentences(TEXT)
    assert s[bm25_rank("retrieval generation grounding", s)[0][0]].startswith("Retrieval")


def test_retry_recovers_and_gives_up():
    n = {"c": 0}
    def flaky():
        n["c"] += 1
        if n["c"] < 3:
            raise IOError("x")
        return "ok"
    assert retry(flaky, base=0.01) == "ok"
    with pytest.raises(IOError):
        retry(lambda: (_ for _ in ()).throw(IOError("x")), attempts=2, base=0.01)


def test_tool_permissions():
    with pytest.raises(PermissionError):
        tools.call("synthesizer", "web_search", "x")


def _ctx():
    b = Blackboard("what is rag")
    b.topic = "retrieval augmented generation"
    s = Source("S1", "http://x", "X", origin="web", text=TEXT)
    b.sources["S1"] = s
    b.evidence = [Evidence("E1", "S1", "Q1", sentences(TEXT)[0], 1.0)]
    return agents.Ctx(OfflineLLM(), b, lambda *a, **k: None)


def test_citer_verifies_and_drops_hallucination():
    ctx = _ctx()
    good = Claim("Retrieval-augmented generation combines a language model with an external retriever", ["E1"])
    bad = Claim("Quantum entanglement powers the retriever across galaxies", ["E1"])
    nocite = Claim("Something unsupported entirely", [])
    out = agents.citer(ctx, [good, bad, nocite], [Section("A", [good])])
    assert out["stats"]["dropped"] == 2 and out["summary"][0]["cites"] == [1]


def test_citer_rejects_fabricated_quote():
    ctx = _ctx()
    ctx.board.evidence[0].quote = "This quote does not occur in the source text at all, honestly."
    out = agents.citer(ctx, [Claim("This quote does not occur in the source text", ["E1"])], [])
    assert out["stats"]["verified"] == 0


def test_run_fails_gracefully(monkeypatch, tmp_path):
    monkeypatch.setenv("CORTEX_RUNS_DIR", str(tmp_path))
    def boom(*a, **k): raise IOError("offline")
    monkeypatch.setitem(tools.TOOLS, "wikipedia_search", boom)
    monkeypatch.setitem(tools.TOOLS, "web_search", boom)
    monkeypatch.setitem(tools.TOOLS, "arxiv_search", boom)
    monkeypatch.setattr("cortex_research.resilience.time.sleep", lambda s: None)
    r = Run("what is retrieval augmented generation", "quick", None)
    execute(r, OfflineLLM())
    assert r.status == "failed" and "No sources" in r.error


def test_full_pipeline_with_fake_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("CORTEX_RUNS_DIR", str(tmp_path))
    monkeypatch.setitem(tools.TOOLS, "wikipedia_search", lambda q, n=4: [{"title": "RAG", "url": "http://w/RAG", "snippet": "", "origin": "wikipedia"}])
    monkeypatch.setitem(tools.TOOLS, "web_search", lambda q, n=4: (_ for _ in ()).throw(IOError("ddg down")))
    monkeypatch.setitem(tools.TOOLS, "arxiv_search", lambda q, n=3: [])
    monkeypatch.setitem(tools.TOOLS, "wikipedia_text", lambda t: TEXT * 3)
    monkeypatch.setattr("cortex_research.resilience.time.sleep", lambda s: None)
    r = Run("explain retrieval augmented generation", "quick", "sess")
    execute(r, OfflineLLM())
    assert r.status == "done", r.error
    assert r.report["stats"]["grounding"] == 1.0
    assert any(e["type"] == "retry" for e in r.events) and any(e["type"] == "warn" for e in r.events)
    assert "## References" in r.report["markdown"]


def test_verifier_catches_meaning_flips():
    from cortex_research import verifier
    if verifier.load() is None:
        pytest.skip("model not trained")
    q = "The reactor can sustain a plasma temperature of 150 million degrees using superconducting magnets."
    assert verifier.score(q, [q])[0] > 0.9
    assert verifier.score("The reactor cannot sustain a plasma temperature of 150 million degrees using superconducting magnets.", [q])[0] < 0.5
    assert verifier.score("The reactor can sustain a plasma temperature of 900 million degrees using superconducting magnets.", [q])[0] < 0.5


def test_critic_finds_gaps():
    ctx = _ctx()
    ctx.board.subqs = [agents.SubQuestion("Q1", "Overview", "x"), agents.SubQuestion("Q2", "Challenges & limitations", "y")]
    assert [g.id for g in agents.critic_gaps(ctx, min_evidence=1)] == ["Q2"]
