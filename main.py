"""Command line entry point for Self-Correcting Multi-Agent RAG.

Usage:
    python main.py "When did the Yavuz Sultan Selim Bridge open?"
    python main.py "What is the capital of Turkey?" --docs documents.txt
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from rag.pipeline import run_pipeline

SAMPLE_DOCS = """The Bosphorus connects the Black Sea to the Sea of Marmara and is approximately 31 kilometers long.
The Fatih Sultan Mehmet Bridge opened in 1988 and is the second bridge over the Bosphorus.
The 15 July Martyrs Bridge opened in 1973 as the first suspension bridge over the Bosphorus.
The Yavuz Sultan Selim Bridge opened in 2016, becoming the third bridge over the Bosphorus.
Marmaray is an undersea railway tube tunnel crossing beneath the Bosphorus, opened in 2013.
The Eurasia Tunnel opened in 2016 to serve vehicular road traffic under the strait.
Cappadocia is a historic region known for its fairy chimneys and hot air balloon tours.
The capital of Turkey is Ankara; it was officially declared the capital on October 13, 1923.
Lake Van is the largest lake in Turkey and is a saline soda lake.
Mount Ararat, with an elevation of 5,137 meters, is the highest peak in Turkey."""


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Multi-Agent Self-Correcting RAG Pipeline")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--docs", help="Document file path (one chunk per line). Uses built-in sample docs if omitted.")
    parser.add_argument("--model", default=os.getenv("MODEL", "gpt-4o-mini"), help="Model identifier")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks retrieved initially")
    parser.add_argument("--max-retries", type=int, default=1, help="Max re-retrieval attempts when results are insufficient")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not defined. Please add it to your .env file.")

    docs = open(args.docs, encoding="utf-8").read() if args.docs else SAMPLE_DOCS

    result = run_pipeline(
        args.question,
        docs,
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        model=args.model,
        k=args.k,
        max_retries=args.max_retries,
    )

    print("\n=== Agent Log ===")
    for line in result.log:
        print(" ", line)

    print("\n=== Grader Evaluation ===")
    for g in result.grade.grades:
        print(f"  {g.chunk_id}: {g.label} — {g.reason}")

    print("\n=== Answer ===")
    print(result.answer)

    print("\n=== Cost & Performance ===")
    print(f"  LLM calls: {result.llm_calls} | Tokens: {result.total_tokens} | Elapsed: {result.elapsed_ms} ms | Retries: {result.retries}")


if __name__ == "__main__":
    main()
