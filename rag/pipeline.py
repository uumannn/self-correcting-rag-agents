"""Three-agent self-correcting RAG pipeline.

Flow:
1. Retrieval Agent  -> selects the top-k most similar chunks from documents (no LLM, stem matching)
2. Grader Agent     -> labels each chunk as RELEVANT / IRRELEVANT / AMBIGUOUS;
                       triggers re-retrieval if chunks are insufficient (CRAG-style correction)
3. Answer Agent     -> generates a response using only RELEVANT chunks;
                       explicitly states when information is missing (no hallucinations)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import OpenAI

from .agents import GradeResult, answer, grade_chunks
from .retriever import Chunk, retrieve


@dataclass
class PipelineResult:
    question: str
    answer: str
    chunks: list[Chunk]
    grade: GradeResult
    llm_calls: int
    total_tokens: int
    elapsed_ms: int
    retries: int
    log: list[str] = field(default_factory=list)


def run_pipeline(
    question: str,
    docs: str,
    *,
    api_key: str,
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
    k: int = 4,
    max_retries: int = 1,
) -> PipelineResult:
    client = OpenAI(api_key=api_key, base_url=base_url)
    log: list[str] = []
    llm_calls = 0
    total_tokens = 0
    start = time.time()

    # 1) Retrieval Agent
    chunks = retrieve(docs, question, k=k)
    log.append(f"[Retrieval] {len(chunks)} chunks selected: " + ", ".join(f"{c.id}({c.score:.2f})" for c in chunks))

    # 2) Grader Agent (+ re-retrieval if needed)
    retries = 0
    grade = grade_chunks(client, model, question, chunks)
    llm_calls += 1
    total_tokens += grade.usage.get("prompt_tokens", 0) + grade.usage.get("completion_tokens", 0)
    log.append(f"[Grader] Decision: {'SUFFICIENT' if grade.sufficient else 'INSUFFICIENT'}")

    while not grade.sufficient and retries < max_retries:
        retries += 1
        log.append(f"[Correction] Re-retrieval attempt {retries}/{max_retries} (wider search)")
        chunks = retrieve(docs, question, k=k * 2)  # CRAG-style: expand scope
        grade = grade_chunks(client, model, question, chunks)
        llm_calls += 1
        total_tokens += grade.usage.get("prompt_tokens", 0) + grade.usage.get("completion_tokens", 0)
        log.append(f"[Grader] New decision: {'SUFFICIENT' if grade.sufficient else 'INSUFFICIENT'}")

    relevant_ids = {g.chunk_id for g in grade.grades if g.label == "RELEVANT"}
    relevant = [c for c in chunks if c.id in relevant_ids]
    log.append(f"[Grader] Relevant chunks: {', '.join(c.id for c in relevant) or 'none'}")

    # 3) Answer Agent
    final_answer, usage = answer(client, model, question, relevant)
    llm_calls += 1
    total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    elapsed_ms = int((time.time() - start) * 1000)
    return PipelineResult(
        question=question,
        answer=final_answer,
        chunks=chunks,
        grade=grade,
        llm_calls=llm_calls,
        total_tokens=total_tokens,
        elapsed_ms=elapsed_ms,
        retries=retries,
        log=log,
    )
