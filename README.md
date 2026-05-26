# Financial-Compliance RAG Q&A System

*A personal project, designed with production thinking — built to explore and demonstrate
RAG patterns for regulated-finance use cases. It is a learning/portfolio project, not a
deployed production system.*

A retrieval-augmented question-answering system over **financial / banking compliance
content**. Users ask product, fees, AML, and data-protection questions; the system
retrieves from a document knowledge base, answers **only from retrieved evidence**
with **inline citations**, and **refuses (instead of hallucinating)** when the answer
isn't in the corpus — with PII masking and output guardrails on the safety layer.

> Focus: grounded responses, auditability (citations), refusal control, PII handling,
> and evaluation — the patterns that matter when RAG meets a regulated domain.

---

## What it does (demo)

**In-corpus question → grounded answer with citations:**
```
Q: What is the cash advance fee on FAB credit cards?
A: The cash advance fee on FAB credit cards is up to 3.15% of the amount of cash
   obtained, with a minimum of AED 157.50 per transaction [4][1].
   Sources: [1] fab-consolidated-credit-cards p.3  [4] Fees-and-charges-FAB p.0
```

**Out-of-corpus question → explicit refusal (no hallucination):**
```
Q: What is the interest rate on a Tesla car loan in Japan?
A: I cannot answer this based on the available documents.
   [refused at gate 1 | relevance 0.612 < threshold 0.65]
```

---

## Architecture

**Offline (indexing):**
```
PDFs → per-page load → token-based chunking → BGE embeddings → pgvector
```

**Online (query):**
```
question → retrieve top-k (with scores)
         → GATE 1: relevance-threshold refusal (skip LLM if too low)
         → grounding prompt (only-from-context + inline [n] citations)
         → GATE 2: LLM refusal if context insufficient
         → answer + sources
```

---

## Tech stack & why

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Orchestration | **LangChain (v1)** | Mature RAG ecosystem; fast composition |
| Vector DB | **pgvector** (Postgres 17) | SQL + vectors in one store; realistic for fintech, no extra infra |
| Embeddings | **BGE-base-en-v1.5** (local) | Runs locally → data never leaves the host (compliance); strong MTEB retrieval |
| LLM | **Qwen2.5-7B** via **Ollama** (local) | On-prem/sovereign-cloud friendly; multilingual (incl. Arabic/South-Asian) |
| Chunking | Fixed **500 tokens / 64 overlap** | Token count via the *embedding model's own tokenizer* (see below) |
| PII / Guardrails | **Presidio** (+ custom Emirates ID) · rule-based output guardrails | Input masking + output PII-leak / citation checks |
| Eval | hand-written retrieval metrics + **local LLM-as-judge** | Separate retrieval vs generation diagnosis; fully local |

**Corpus** (public, no real PII): BIS AML/CFT guidelines, EU GDPR, FAB fees schedule,
FAB credit-card key facts, Emirates NBD general T&Cs — **5 docs, 573 chunks**.

---

## Key design decisions & trade-offs

1. **Chunking tokenizer aligned to the embedding model.** Chunk size is measured with
   *BGE's own tokenizer* (`SentenceTransformersTokenTextSplitter`), not characters or
   tiktoken. Why: only then can we guarantee each chunk fits BGE's **512-token limit**
   (set 500 for headroom incl. special tokens); otherwise oversized chunks are silently
   truncated → lost content. Trade-off: chose fixed-size over recursive/semantic for a
   clean baseline; chunk size is a planned A/B experiment.

2. **Embeddings normalized for cosine.** `normalize_embeddings=True` — with cosine
   similarity, vectors must be unit-length or scores are wrong. (BGE-v1.5 also reduces
   reliance on the `query:` instruction prefix, so it's omitted for simplicity.)

3. **Refusal = two gates (defense in depth).**
   - **Gate 1 — relevance threshold:** if the top retrieval score is too low, refuse
     *without calling the LLM* (cheap, deterministic, zero hallucination risk).
   - **Gate 2 — grounding prompt:** the LLM is instructed to use *only* the context and
     emit a fixed refusal string if it's insufficient (catches "retrieved but not enough").
   - Why both: a model refusing on its own is luck, not a guarantee.

4. **The threshold was calibrated, not guessed.** Measured score distributions:
   in-corpus **0.716–0.763** vs out-of-corpus **0.426–0.612** → set threshold **0.65**.
   Notable finding: a *finance-flavored* out-of-corpus question ("Tesla car loan",
   **0.612**) landed only ~0.1 below in-corpus — a **thin margin** that proves a
   threshold alone is fragile and justifies Gate 2.

5. **Auditability via per-page citations.** PDFs are loaded per page so every chunk
   carries `(doc, page)`; answers cite inline `[n]` mapped to source + page — the
   "audit-ready / grounded" requirement.

6. **Local-first for compliance.** Embeddings and LLM both run locally (BGE + Ollama),
   so document and query data never leave the machine — the on-prem/sovereign-cloud
   narrative a regulated fintech cares about.

7. **PII masked at the entry point.** A user query is PII-masked (Microsoft Presidio +
   a **custom Emirates ID recognizer**) *before* it touches logs, retrieval, or the LLM,
   so every downstream component only ever sees the redacted version. Detection is scoped
   to a whitelist of banking-relevant entity types to control false positives (see gotcha).

8. **Output guardrails make grounding enforceable.** Every non-refusal answer must pass
   (a) a PII-leak scan and (b) **inline-citation enforcement** — an answer with no `[n]`
   citation is treated as ungrounded and blocked. Turns "be grounded" into a checkable rule.

### Gotchas solved along the way
- **PII detector false positives:** the small spaCy model tagged "Emirates" as a LOCATION
  (so "Emirates ID" got over-masked). Fixed by **whitelisting** the entity types we mask
  (cards, email, phone, IBAN, names, Emirates ID) and excluding LOCATION/DATE/URL — a
  precision/recall call that preserves query meaning.
- **pgvector ↔ Postgres version match:** Homebrew's `pgvector` ships the extension only
  for PG17/18; installing PG16 first failed with "extension vector is not available" →
  switched to PG17.
- **Inline-XBRL noise (earlier SEC-filing prototype):** modern filings are inline XBRL;
  a naive `.get_text()` dumps the hidden `<ix:header>` block into the body → fixed by
  stripping hidden XBRL nodes while keeping the visible numbers.
- **Vector index at scale:** at 573 chunks pgvector uses exact search (fine/instant);
  for scale, add an **HNSW** index — a deliberate "not yet needed" call.

---

## Components

- **Indexing pipeline** (`load.py`, `build_index.py`) — per-page PDF load → token-based chunking → BGE embeddings → pgvector.
- **Grounded retrieval & generation** (`rag.py`) — two-gate refusal, inline citations, calibrated relevance threshold.
- **PII masking & output guardrails** (`pii.py`, `guardrails.py`, `pipeline.py`) — Presidio + custom Emirates ID recognizer on input; PII-leak scan + citation enforcement on output; composed end-to-end in `safe_answer()`.
- **Evaluation harness** (`eval_data.py`, `evaluate.py`) — retrieval hit-rate/MRR, refusal accuracy, faithfulness (local LLM-as-judge), and top-k / threshold ablations (results below).

## Evaluation results

20-question hand-labeled test set (15 in-corpus with gold source docs, 5 out-of-corpus refusal cases).

| Metric | Score |
|--------|-------|
| Retrieval **hit@5** | **93.3%** (14/15), MRR 0.933 |
| **Refusal accuracy** @ threshold 0.65 | **95%** (19/20) |
| Generation **faithfulness** (local LLM-as-judge) | **92.9%** (13/14 answered) |

**Ablation — refusal threshold sweep** (the headline finding):

| threshold | 0.55 | 0.60 | **0.65** | 0.70 | 0.75 |
|-----------|------|------|----------|------|------|
| refusal accuracy | 90% | 95% | **95%** | 90% | 65% |

→ Accuracy peaks at **0.60–0.65** and degrades on both sides: too low fails to refuse
finance-flavored out-of-corpus questions; too high wrongly refuses valid ones. Empirically
validates the calibrated threshold and demonstrates the precision/recall trade-off.

**Ablation — top-k:** hit@3 = hit@5 = 93.3%; hit@10 = 100% (MRR 0.933 → 0.944). k=10
recovers the one miss but feeds noisier context to the LLM → kept **k=5** as the balance.

---

## Setup & run

```bash
# Postgres 17 + pgvector + Ollama (macOS / Homebrew)
brew install postgresql@17 pgvector ollama
brew services start postgresql@17 && brew services start ollama
createdb financial_rag
psql financial_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
ollama pull qwen2.5:7b

# Python
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m spacy download en_core_web_sm   # for Presidio NER

# Put PDFs in data/raw/docs/, then build the index:
./.venv/bin/python -m src.build_index

# Ask via the full safety pipeline (input PII masking → grounded RAG → output guardrails):
./.venv/bin/python -m src.pipeline "What is the cash advance fee on FAB credit cards?"

# Run the evaluation harness (retrieval / refusal / faithfulness + ablations):
./.venv/bin/python -m src.evaluate
```

## Known limitations / future work
- Tables are flattened to text (numeric questions weaker); planned upgrade: structured
  table/financial-data handling + hybrid (dense + BM25) retrieval and a reranker.
- Threshold calibrated on a small (20-question) sample; a larger labeled set would refine it.
- Single-turn only; conversational rewriting is a future extension.
