"""End-to-end evaluation: runs the full pipeline on a question set and reports quality + reliability.
Usage: python -m eval.run_eval [n_questions]   → eval/results.json"""
import json
import statistics as st
import sys
import time
from cortex_research.llm import OfflineLLM
from cortex_research.orchestrator import Run, execute

QUESTIONS = [
    "How does retrieval-augmented generation reduce LLM hallucinations?",
    "What are the main challenges of nuclear fusion power?",
    "How do mRNA vaccines work and what are their limitations?",
    "What is the current state of quantum error correction?",
    "How does CRISPR gene editing work and what are the ethical concerns?",
    "What causes antibiotic resistance and how can it be slowed?",
    "How do transformer neural networks process language?",
    "What are the trade-offs of solar power versus wind power?",
]


def main(n=len(QUESTIONS)):
    rows = []
    for q in QUESTIONS[:n]:
        r = Run(q, "standard", None)
        t = time.time()
        execute(r, OfflineLLM())
        rep = r.report or {}
        stt = rep.get("stats", {})
        rows.append({"question": q, "status": r.status, "seconds": round(time.time() - t, 1),
                     "claims": stt.get("claims", 0), "verified": stt.get("verified", 0), "grounding": stt.get("grounding", 0),
                     "sources_cited": stt.get("sources_cited", 0), "sections": len(rep.get("sections", [])),
                     "retries": sum(e["type"] == "retry" for e in r.events), "warnings": sum(e["type"] == "warn" for e in r.events),
                     "critic_rounds": sum(1 for e in r.events if e.get("node") == "critic" and "second search" in e.get("message", ""))})
        print(rows[-1])
    ok = [r for r in rows if r["status"] == "done"]
    summary = {"runs": len(rows), "success_rate": round(len(ok) / len(rows), 2),
               "median_seconds": st.median(r["seconds"] for r in ok) if ok else None,
               "mean_grounding": round(st.mean(r["grounding"] for r in ok), 3) if ok else None,
               "mean_sources_cited": round(st.mean(r["sources_cited"] for r in ok), 1) if ok else None,
               "total_retries_absorbed": sum(r["retries"] for r in rows), "total_warnings_survived": sum(r["warnings"] for r in rows)}
    json.dump({"summary": summary, "runs": rows}, open("eval/results.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else len(QUESTIONS))
