"""Learned claim verifier: does `quote` actually support `claim`?

A gradient-boosted classifier over hand-engineered alignment features. It catches failures that
plain word-overlap misses: negation flips, antonym swaps, changed numbers, swapped entities and
unsupported add-on clauses. Trained by `python -m cortex_research.train_verifier`.
"""
from __future__ import annotations
import difflib
import json
import os
import re
from .textutil import tokens

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "verifier.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "verifier_metrics.json")

NEG = {"not", "no", "never", "cannot", "without", "neither", "nor", "none", "fails", "unable", "isn't", "doesn't", "don't", "can't", "won't"}
ANTONYMS = [("increase", "decrease"), ("increases", "decreases"), ("higher", "lower"), ("more", "less"), ("improve", "worsen"),
            ("improves", "worsens"), ("reduce", "increase"), ("reduces", "increases"), ("faster", "slower"), ("larger", "smaller"),
            ("before", "after"), ("first", "last"), ("allow", "prevent"), ("allows", "prevents"), ("enable", "disable"),
            ("enables", "disables"), ("support", "oppose"), ("supports", "opposes"), ("gain", "loss"), ("rise", "fall")]
_ANT = {}
for a, b in ANTONYMS:
    _ANT.setdefault(a, set()).add(b)
    _ANT.setdefault(b, set()).add(a)

FEATURES = ["prec", "rec", "bigram_prec", "seq_ratio", "len_ratio", "num_new", "num_miss", "ent_new", "neg_diff",
            "antonym_flip", "new_content", "new_content_frac", "contains"]


def _words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'\-]+", s.lower())


def _nums(s: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def _ents(s: str) -> set[str]:
    return {w.lower() for w in re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-zA-Z]{2,}\b", s)}


def features(claim: str, quote: str) -> list[float]:
    ct, qt = tokens(claim), tokens(quote)
    cs, qs = set(ct), set(qt)
    prec = len(cs & qs) / len(cs) if cs else 0.0
    rec = len(cs & qs) / len(qs) if qs else 0.0
    cb = set(zip(ct, ct[1:]))
    qb = set(zip(qt, qt[1:]))
    bigram = len(cb & qb) / len(cb) if cb else prec
    cw, qw = _words(claim), _words(quote)
    seq = difflib.SequenceMatcher(None, cw, qw, autojunk=False).ratio()
    nc, nq = _nums(claim), _nums(quote)
    ec, eq = _ents(claim), _ents(quote)
    neg = len(NEG & set(cw)) - len(NEG & set(qw))
    flip = 0
    for w in cw:
        if w in _ANT and w not in qw and (_ANT[w] & set(qw)):
            flip += 1
    new = cs - qs
    return [prec, rec, bigram, seq, len(cw) / max(1, len(qw)), len(nc - nq), len(nq - nc) if nc else 0,
            len(ec - eq), abs(neg), flip, len(new), len(new) / max(1, len(cs)), float(normalize_in(claim, quote))]


def normalize_in(claim: str, quote: str) -> bool:
    n = lambda s: " ".join(_words(s))
    return n(claim) in n(quote)


_model = None
_loaded = False


def load():
    global _model, _loaded
    if not _loaded:
        _loaded = True
        try:
            import joblib
            _model = joblib.load(MODEL_PATH)
        except Exception:  # noqa: BLE001 - model optional; lexical fallback
            _model = None
    return _model


def info() -> dict:
    try:
        with open(METRICS_PATH) as f:
            return {"loaded": load() is not None, **json.load(f)}
    except Exception:  # noqa: BLE001
        return {"loaded": False}


def score(claim: str, quotes: list[str]) -> tuple[float, str]:
    """Return (support probability 0-1, method). Best-supporting quote wins."""
    m = load()
    if m is None:
        ct = set(tokens(claim))
        pool = {t for q in quotes for t in tokens(q)}
        return (round(len(ct & pool) / len(ct), 2) if ct else 0.0), "lexical"
    X = [features(claim, q) for q in quotes]
    return round(float(max(m.predict_proba(X)[:, 1])), 3), "learned"
