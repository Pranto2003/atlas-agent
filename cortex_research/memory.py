"""Memory: per-run blackboard (shared state between agents) + per-session conversation memory."""
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from .models import Source, Evidence
from .textutil import bm25_rank


@dataclass
class Blackboard:
    question: str
    topic: str = ""
    subqs: list = field(default_factory=list)
    sources: dict[str, Source] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_source(self, s: Source) -> bool:
        with self.lock:
            if any(x.url == s.url for x in self.sources.values()):
                return False
            self.sources[s.id] = s
            return True


class SessionMemory:
    """Remembers earlier turns so follow-up questions reuse prior evidence."""
    def __init__(self):
        self._s: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def turns(self, sid: str) -> list[dict]:
        with self._lock:
            return list(self._s.get(sid, []))

    def remember(self, sid: str, question: str, summary: str, evidence: list[dict]):
        with self._lock:
            self._s.setdefault(sid, []).append({"question": question, "summary": summary, "evidence": evidence})
            self._s[sid] = self._s[sid][-8:]

    def recall(self, sid: str, query: str, k: int = 6) -> list[dict]:
        pool = [e for t in self.turns(sid) for e in t["evidence"]]
        if not pool:
            return []
        ranked = bm25_rank(query, [e["quote"] for e in pool])
        return [pool[i] for i, s in ranked[:k] if s > 0]
