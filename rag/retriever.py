"""Retrieval Agent

Lightweight stem-based scoring agent without external AI dependencies.
Splits documents into chunks and ranks them based on prefix/stem overlap
with the query.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    score: float


def _normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    words = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).split()
    return [w for w in words if len(w) > 2]


def retrieve(docs: str, query: str, k: int = 4) -> list[Chunk]:
    """Splits document text into lines and returns the top-k most relevant chunks."""
    chunks = [line.strip() for line in docs.splitlines() if line.strip()]
    query_stems = [w[:5] for w in _normalize(query)]

    scored: list[Chunk] = []
    for i, text in enumerate(chunks):
        stems = {w[:5] for w in _normalize(text)}
        hits = sum(1 for s in query_stems if s in stems)
        score = hits / max(len(query_stems), 1)
        scored.append(Chunk(id=f"P{i + 1}", text=text, score=score))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]
