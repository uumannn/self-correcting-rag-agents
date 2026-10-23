"""Grader and Answer agents (LLM-based)."""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from .retriever import Chunk

GRADER_SYS = """You are a retrieval grader agent. Evaluate each chunk against the given question.
Format each chunk evaluation as: <ID>: RELEVANT | IRRELEVANT | AMBIGUOUS — short explanation
End with a single final line: DECISION: SUFFICIENT or DECISION: INSUFFICIENT"""

ANSWER_SYS = """You are a synthesized answer agent. Based ONLY on the provided RELEVANT chunks, formulate a concise and factual answer.
Cite the chunk identifiers you used, such as [P1] or [P2].
If the chunks do not contain enough information to answer the question, clearly state: "I do not have enough information to answer this question." and briefly explain what is missing. Never fabricate or hallucinate facts."""


@dataclass
class Grade:
    chunk_id: str
    label: str  # RELEVANT | IRRELEVANT | AMBIGUOUS
    reason: str


@dataclass
class GradeResult:
    grades: list[Grade]
    sufficient: bool
    raw: str
    usage: dict = field(default_factory=dict)


def grade_chunks(client: OpenAI, model: str, question: str, chunks: list[Chunk]) -> GradeResult:
    """Grader Agent: labels each retrieved chunk based on its relevance."""
    listing = "\n".join(f"{c.id}: {c.text}" for c in chunks)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GRADER_SYS},
            {"role": "user", "content": f"Question: {question}\n\nChunks:\n{listing}"},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""

    grades: list[Grade] = []
    sufficient = False
    for line in raw.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("DECISION:") or upper.startswith("KARAR:"):
            sufficient = ("SUFFICIENT" in upper or "YETERLİ" in upper or "YETERLI" in upper) and (
                "INSUFFICIENT" not in upper and "YETERSİZ" not in upper and "YETERSIZ" not in upper
            )
            continue
        if ":" in line:
            cid, rest = line.split(":", 1)
            cid = cid.strip()
            rest_upper = rest.upper()
            label = (
                "IRRELEVANT" if ("IRRELEVANT" in rest_upper or "ALAKASIZ" in rest_upper)
                else "AMBIGUOUS" if ("AMBIGUOUS" in rest_upper or "BELİRSİZ" in rest_upper or "BELIRSIZ" in rest_upper)
                else "RELEVANT" if ("RELEVANT" in rest_upper or "ALAKALI" in rest_upper)
                else "AMBIGUOUS"
            )
            delimiter = "—" if "—" in rest else "-" if "-" in rest else ":"
            reason = rest.split(delimiter, 1)[-1].strip() if delimiter in rest else rest.strip()
            grades.append(Grade(chunk_id=cid, label=label, reason=reason))

    usage = {}
    if resp.usage:
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
    return GradeResult(grades=grades, sufficient=sufficient, raw=raw, usage=usage)


def answer(
    client: OpenAI,
    model: str,
    question: str,
    relevant: list[Chunk],
    previous_answer: str | None = None,
) -> tuple[str, dict]:
    """Answer Agent: generates a grounded response using only relevant chunks."""
    listing = "\n".join(f"{c.id}: {c.text}" for c in relevant) or "(no relevant chunks)"
    user = f"Question: {question}\n\nRelevant chunks:\n{listing}"
    if previous_answer:
        user += f"\n\nPrevious (insufficient) answer:\n{previous_answer}\nPlease correct and complete this answer."

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_SYS},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    usage = {}
    if resp.usage:
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
    return (resp.choices[0].message.content or "").strip(), usage
