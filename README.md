# Self-Correcting Multi-Agent RAG

Standard RAG hands every retrieved chunk to the LLM and hopes for the best.  
This pipeline adds a **Grader agent** in the middle that reads each chunk, decides whether it actually answers the question, and — if the batch is too weak — triggers a wider re-retrieval automatically before the Answer agent ever runs.

No hallucinations. Cited answers. Transparent cost log per query.

---

## Architecture

```
Question
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  Retrieval Agent  (retriever.py)                    │
│  • Splits docs into chunks, scores by stem overlap  │
│  • Pure Python — zero LLM calls, zero latency cost  │
└────────────────────────┬────────────────────────────┘
                         │  top-k chunks
                         ▼
┌─────────────────────────────────────────────────────┐
│  Grader Agent  (agents.py)                          │
│  • Labels each chunk: RELEVANT / IRRELEVANT /       │
│    AMBIGUOUS                                        │
│  • If verdict is INSUFFICIENT → doubles k,          │
│    re-fetches, grades again  (CRAG loop)            │
└────────────────────────┬────────────────────────────┘
                         │  verified relevant chunks
                         ▼
┌─────────────────────────────────────────────────────┐
│  Answer Agent  (agents.py)                          │
│  • Answers using RELEVANT chunks only, cites [P#]   │
│  • If nothing relevant exists → says so clearly     │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
               Answer + cost report
```

---

## Setup

```bash
git clone https://github.com/uumannn/self-correcting-rag-agents.git
cd self-correcting-rag-agents

pip install -r requirements.txt

# Copy the env template and add your OpenAI key
cp .env.example .env
```

`.env`:
```
OPENAI_API_KEY=sk-...
```

---

## Usage

```bash
# Ask something against the built-in Turkish geography sample docs
python main.py "When did the Yavuz Sultan Selim Bridge open?"

# Use your own document file (plain text, one chunk per line)
python main.py "Who founded the company?" --docs company_wiki.txt

# Tweak retrieval depth and correction budget
python main.py "Your question" --k 8 --max-retries 2 --model gpt-4o
```

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--k` | `4` | Chunks fetched on first retrieval pass |
| `--max-retries` | `1` | How many times the CRAG loop may widen the search |
| `--model` | `gpt-4o-mini` | Any OpenAI-compatible model name |
| `--docs` | built-in sample | Path to your own document file |

---

## Sample Output

```
=== Agent Log ===
  [Retrieval] 4 chunks selected: P4(0.67), P1(0.33), P2(0.17), P3(0.00)
  [Grader] Decision: SUFFICIENT
  [Grader] Relevant chunks: P4, P1

=== Grader Evaluation ===
  P4: RELEVANT — "opened in 2016, third bridge over the strait"
  P1: RELEVANT — confirms Yavuz Sultan Selim name and year
  P2: IRRELEVANT — about a different infrastructure project
  P3: IRRELEVANT — unrelated geographical fact

=== Answer ===
  The Yavuz Sultan Selim Bridge opened in 2016 [P4][P1].

=== Cost ===
  LLM calls: 2 | Tokens: 389 | Time: 761 ms | Retries: 0
```

---

## Files

```
rag/
  retriever.py   — stem-based scoring, no external dependencies
  agents.py      — Grader + Answer agents, structured LLM prompts
  pipeline.py    — orchestrates the three agents, owns the CRAG loop
main.py          — CLI, argument parsing, prints the final report
.env.example     — copy to .env and fill in your key
requirements.txt — openai>=1.40, python-dotenv>=1.0
```
