"""Train + evaluate the claim verifier.

Data: real Wikipedia sentences across diverse topics. For each sentence we synthesise
  positives  — verbatim, first clause, parenthetical-stripped
  negatives  — negation flip, antonym swap, number change, entity swap, off-topic sentence,
               and 'supported + unsupported add-on clause'
Split is by ARTICLE (no leakage). Reports held-out metrics vs the old word-overlap rule.
Usage: python -m cortex_research.train_verifier
"""
from __future__ import annotations
import json
import random
import re
import time
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from . import tools
from .textutil import sentences, tokens
from .verifier import FEATURES, MODEL_PATH, METRICS_PATH, _ANT, features

TOPICS = ["Retrieval-augmented generation", "Large language model", "Nuclear fusion", "mRNA vaccine", "Quantum computing",
          "Photosynthesis", "Black hole", "Blockchain", "Climate change", "CRISPR", "Transformer (deep learning architecture)",
          "Solar power", "Antibiotic resistance", "Machine learning", "Plate tectonics", "Internet of things", "Cryptography",
          "Neural network", "Electric vehicle", "Gene therapy", "General relativity", "Supply chain", "Inflation", "Operating system",
          "Cloud computing", "Reinforcement learning", "Superconductivity", "Immune system", "Renewable energy", "Cybersecurity",
          "Computer vision", "Natural language processing", "Semiconductor", "Vaccination", "Ocean acidification", "Graph theory",
          "Database", "Evolution", "Microbiome", "Robotics", "Cosmic microwave background", "Diabetes", "Printing press",
          "Industrial Revolution", "Artificial neural network", "Search engine", "Compiler", "Telescope", "Volcano", "Genome"]
rng = random.Random(7)


def negate(s):
    for a, b in [(r"\bis\b", "is not"), (r"\bare\b", "are not"), (r"\bcan\b", "cannot"), (r"\bwas\b", "was not"),
                 (r"\bhas\b", "has not"), (r"\bwere\b", "were not"), (r"\bwill\b", "will not")]:
        if re.search(a, s):
            return re.sub(a, b, s, count=1)
    return None


def antonym(s):
    ws = s.split()
    idx = [i for i, w in enumerate(ws) if w.lower().strip(",.;") in _ANT]
    if not idx:
        return None
    i = rng.choice(idx)
    w = ws[i].strip(",.;")
    rep = rng.choice(sorted(_ANT[w.lower()]))
    ws[i] = ws[i].replace(w, rep.capitalize() if w[0].isupper() else rep)
    return " ".join(ws)


def change_number(s):
    m = list(re.finditer(r"\d+(?:\.\d+)?", s))
    if not m:
        return None
    x = rng.choice(m)
    v = float(x.group())
    new = str(int(v * rng.choice([2, 3, 5, 10]) + rng.randint(1, 9))) if v == int(v) else str(round(v * 3.7, 1))
    return s[:x.start()] + new + s[x.end():]


def swap_entity(s, pool):
    ents = list(re.finditer(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]{3,}\b", s))
    if not ents:
        return None
    x = rng.choice(ents)
    rep = rng.choice(pool)
    return s[:x.start()] + rep + s[x.end():] if rep.lower() != x.group().lower() else None


def first_clause(s):
    parts = re.split(r",|;| — | - |:", s)
    return parts[0].strip() if len(parts[0].split()) >= 6 else None


def strip_paren(s):
    t = re.sub(r"\s*\([^)]*\)", "", s)
    return t if t != s else None


def build_pairs(sents, other_sents, ent_pool):
    pos, neg = [], []
    for s in sents:
        pos.append((s, s, "verbatim"))
        for fn, tag in ((first_clause, "clause"), (strip_paren, "paren")):
            c = fn(s)
            if c:
                pos.append((c, s, tag))
        for fn, tag in ((negate, "negation"), (antonym, "antonym"), (change_number, "number")):
            c = fn(s)
            if c and c != s:
                neg.append((c, s, tag))
        c = swap_entity(s, ent_pool)
        if c:
            neg.append((c, s, "entity"))
        neg.append((rng.choice(sents), s, "same-topic-other"))
        if other_sents:
            neg.append((rng.choice(other_sents), s, "off-topic"))
            neg.append((s.rstrip(".") + ", and " + rng.choice(other_sents)[0].lower() + rng.choice(other_sents)[1:], s, "addon"))
    return pos, neg


def fetch_corpus():
    corpus = {}
    for t in TOPICS:
        try:
            ss = sentences(tools.wikipedia_text(t))
            if len(ss) >= 15:
                corpus[t] = ss[:60]
        except Exception as e:  # noqa: BLE001
            print("skip", t, type(e).__name__)
    return corpus


def main():
    t0 = time.time()
    corpus = fetch_corpus()
    print(f"corpus: {len(corpus)} articles, {sum(map(len, corpus.values()))} sentences")
    topics = list(corpus)
    rng.shuffle(topics)
    cut = int(len(topics) * 0.75)
    splits = {"train": topics[:cut], "test": topics[cut:]}
    ent_pool = sorted({w for ss in corpus.values() for s in ss for w in re.findall(r"\b[A-Z][a-z]{3,}\b", s)})[:800]
    data = {}
    for name, ts in splits.items():
        X, y, tags = [], [], []
        for t in ts:
            others = [s for u in ts if u != t for s in corpus[u][:10]]
            pos, neg = build_pairs(corpus[t], others, ent_pool)
            for c, q, tag in pos:
                X.append(features(c, q)); y.append(1); tags.append(tag)
            for c, q, tag in neg:
                X.append(features(c, q)); y.append(0); tags.append(tag)
        data[name] = (np.array(X), np.array(y), np.array(tags))
        print(name, data[name][0].shape, "pos rate", round(data[name][1].mean(), 2))
    Xtr, ytr, _ = data["train"]
    Xte, yte, tte = data["test"]
    model = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.08, subsample=0.9, random_state=7)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    # baseline = the original rule: >=45% of claim tokens appear in the quote
    base_pred = (Xte[:, FEATURES.index("prec")] >= 0.45).astype(int)
    base_score = Xte[:, FEATURES.index("prec")]
    per_type = {}
    for tag in sorted(set(tte)):
        m = tte == tag
        per_type[tag] = {"n": int(m.sum()),
                         "learned_acc": round(float(accuracy_score(yte[m], pred[m])), 3),
                         "baseline_acc": round(float(accuracy_score(yte[m], base_pred[m])), 3)}
    metrics = {
        "trained_at": time.strftime("%Y-%m-%d"), "model": "GradientBoostingClassifier(250 trees)", "features": FEATURES,
        "articles": len(corpus), "train_pairs": int(len(ytr)), "test_pairs": int(len(yte)), "split": "by article (no leakage)",
        "learned": {"auc": round(float(roc_auc_score(yte, p)), 3), "accuracy": round(float(accuracy_score(yte, pred)), 3),
                    "precision": round(float(precision_score(yte, pred)), 3), "recall": round(float(recall_score(yte, pred)), 3),
                    "f1": round(float(f1_score(yte, pred)), 3)},
        "baseline_overlap": {"auc": round(float(roc_auc_score(yte, base_score)), 3), "accuracy": round(float(accuracy_score(yte, base_pred)), 3),
                             "precision": round(float(precision_score(yte, base_pred)), 3), "recall": round(float(recall_score(yte, base_pred)), 3),
                             "f1": round(float(f1_score(yte, base_pred)), 3)},
        "per_error_type": per_type,
        "feature_importance": {f: round(float(v), 3) for f, v in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])},
    }
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ("learned", "baseline_overlap", "per_error_type")}, indent=1))
    print(f"done in {time.time() - t0:.0f}s → {MODEL_PATH}")


if __name__ == "__main__":
    main()
