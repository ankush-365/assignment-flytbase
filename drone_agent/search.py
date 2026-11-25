"""Semantic search over VLM frame captions and track descriptions."""
from __future__ import annotations

import numpy as np

from .db import Database


def _embedder(cfg):
    from .vlm import Embedder

    return Embedder(cfg.embeddings.model, cfg.vlm.host)


def semantic_search(db: Database, run_id: int, text: str, cfg=None, k: int = 8, kind: str | None = None,
                    embedder=None) -> list[dict]:
    rows, mat = db.embeddings(run_id, kind)
    if mat is None:
        return _keyword_fallback(db, run_id, text, k)
    try:
        emb = embedder or _embedder(cfg)
        qv = emb.query(text)
    except Exception:
        return _keyword_fallback(db, run_id, text, k)
    if qv.shape[0] != mat.shape[1]:
        return _keyword_fallback(db, run_id, text, k)
    norms = np.linalg.norm(mat, axis=1) * (np.linalg.norm(qv) or 1.0)
    scores = mat @ qv / np.where(norms == 0, 1.0, norms)
    order = np.argsort(-scores)[:k]
    return [{**rows[i], "score": float(scores[i])} for i in order]


def _keyword_fallback(db: Database, run_id: int, text: str, k: int) -> list[dict]:
    words = [w for w in text.lower().split() if len(w) > 2]
    if not words:
        return []
    hits = []
    for r in db.q("SELECT id, ts, video_t, scene AS text FROM scenes WHERE run_id=?", (run_id,)):
        s = sum(w in r["text"].lower() for w in words) / len(words)
        if s:
            hits.append({**r, "kind": "scene", "ref_id": r["id"], "score": s})
    for r in db.q("SELECT track_id, first_ts AS ts, first_video_t AS video_t, description AS text FROM tracks"
                  " WHERE run_id=? AND verified=1",
                  (run_id,)):
        s = sum(w in (r["text"] or "").lower() for w in words) / len(words)
        if s:
            hits.append({**r, "kind": "track", "ref_id": r["track_id"], "score": s})
    return sorted(hits, key=lambda h: -h["score"])[:k]
