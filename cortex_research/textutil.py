"""Small, dependency-free text utilities: tokenising, sentence splitting, BM25."""
from __future__ import annotations
import math
import re
from collections import Counter

STOP = set("""a an the and or but if then of to in on at by for with about as is are was were be been being it its this that these
those from into than which who whom whose what when where why how do does did can could should would may might will shall not no
so such via vs also more most very over under between within their there they them his her he she we you i our your have has had
tell me explain describe give summary summarize overview current state latest""".split())

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def sentences(text: str, min_len: int = 50, max_len: int = 420) -> list[str]:
    text = re.sub(r"\[\d+\]|\[citation needed\]", "", text)
    out = []
    for block in re.split(r"\n+", text):
        for s in _SENT_SPLIT.split(block.strip()):
            s = re.sub(r"\s+", " ", s).strip()
            if min_len <= len(s) <= max_len and s.count(" ") >= 7:
                out.append(s)
    return out


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def bm25_rank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    """Return (doc_index, score) sorted by descending BM25 score."""
    q = tokens(query)
    toks = [tokens(d) for d in docs]
    n = len(docs)
    if not n or not q:
        return []
    avg = (sum(len(t) for t in toks) / n) or 1.0
    df = Counter(w for t in toks for w in set(t))
    scores = []
    for i, t in enumerate(toks):
        tf = Counter(t)
        s = 0.0
        for w in set(q):
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avg))
        scores.append((i, s))
    return sorted(scores, key=lambda x: -x[1])
