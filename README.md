# Retrieval-Augmented Generation for Aviation Manuals

This repository contains a notebook-first RAG project for question answering over four Edelweiss Air Airbus A340 manuals. The full workflow lives in [npr_mc_1.ipynb](/Users/stefanbinkert/Documents/FHNW_DS/NPR/NPR_MC_1/npr_mc_1/npr_mc_1.ipynb): document loading, section inference, chunking experiments, retrieval evaluation, generation evaluation, plotting, and qualitative error analysis.

## Warning

This notebook takes a very long time to run end to end. It makes a large number of LLM calls for retrieval judging, generator judging, generation, and qualitative analysis, so a full run can take many hours and incur noticeable API cost.

## Scope

The notebook uses a manually curated evaluation set of 100 question-answer-context triplets and compares retrieval and generation choices on:

- `FCTM`: Flight Crew Techniques Manual
- `OM A`: Operations Manual Part A
- `QRH`: Quick Reference Handbook
- `CSPM`: Cabin Safety Procedures Manual

This is not a packaged application. The notebook is the project.

## Repository Layout

```text
.
|-- npr_mc_1.ipynb
|-- requirements.txt
`-- README.md
```

Expected local folders that are used by the notebook but not committed here:

- `data/`
  - the four PDF manuals
  - `gold_standard_triplets_100_v2.json`
  - `faiss_index/` for the saved best vector index
- `task/`
  - course material PDFs referenced in the write-up

## Workflow

The notebook runs in this order:

1. Load the PDF manuals with `PyPDFLoader`.
2. Infer manual-specific section labels and skip likely table-of-contents pages.
3. Build metadata-aware `Document` objects with `manual`, `section`, and `page`.
4. Load the 100-item gold QA set from `data/gold_standard_triplets_100_v2.json`.
5. Evaluate retrievers with graded relevance and `NDCG@4`.
6. Compare retrieval variants:
   - baseline recursive chunking: `chunk_size=1000`, `chunk_overlap=200`
   - larger chunk size: `2000 / 200`
   - title-based chunking with `unstructured`
   - `text-embedding-3-small` vs `text-embedding-3-large`
   - cross-encoder reranking with `BAAI/bge-reranker-base`
   - LLM-assisted retrieval with a draft answer as the query
7. Evaluate generators with an LLM judge on correctness, groundedness, and completeness.
8. Compare generator context size (`k=4` vs `k=8`).
9. Compare generator models: `gpt-5.4`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`.
10. Summarize the final system with qualitative case labels and representative examples.

## Evaluation

Retriever evaluation uses graded relevance:

- `3`: exact labeled evidence
- `2`: alternative but sufficient evidence
- `1`: relevant but incomplete
- `0`: irrelevant

The reported retrieval score is average `NDCG@4`.

Generator evaluation uses an LLM judge and combines:

```text
0.5 * correctness + 0.4 * groundedness + 0.1 * completeness
```

The notebook also tracks:

- total and average generation time
- estimated input and output tokens
- estimated OpenAI text generation cost

## Current Best Retrieval Setup

Based on the saved notebook outputs, the best retriever is:

- chunking: recursive
- `chunk_size=1000`
- `chunk_overlap=200`
- embeddings: `text-embedding-3-large`
- reranker: `BAAI/bge-reranker-base`
- LLM-assisted retrieval: not kept

Saved retriever results:

| Experiment | Avg NDCG@4 | Notes |
|---|---:|---|
| Baseline | `0.8813` | recursive `1000 / 200`, `text-embedding-3-small` |
| Chunk size `2000 / 200` | `0.8573` | `99` valid evaluations |
| Title chunking | `0.8712` | not kept |
| Large embeddings | `0.9080` | `text-embedding-3-large` |
| Reranker | `0.9269` | final retriever |
| LLM-assisted retrieval | `0.8546` | not kept |

The best FAISS index is saved to `data/faiss_index/`.

## Current Generator State

The notebook keeps the `gpt-5-mini` generator with `k=8` as the working final generator chain for downstream qualitative analysis.

Context-size comparison:

| Setup | Avg score | Correctness | Groundedness | Completeness | Avg time | Estimated cost |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-5-mini`, `k=4` | `1.91` | `1.88` | `1.95` | `1.90` | `5.58s` | `$0.033088` |
| `gpt-5-mini`, `k=8` | `1.94` | `1.90` | `1.98` | `1.96` | `4.58s` | `$0.052982` |

Generator model comparison with the best retriever and best `k`:

| Model | Avg score | Correctness | Groundedness | Completeness | Avg time | Estimated cost |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-5.4` | `1.94` | `1.91` | `1.97` | `1.92` | `2.40s` | `$0.504968` |
| `gpt-5` | `1.92` | `1.89` | `1.96` | `1.92` | `8.15s` | `$0.256764` |
| `gpt-5-mini` | `1.94` | `1.90` | `1.98` | `1.96` | `4.58s` | `$0.052982` |
| `gpt-5-nano` | `1.82` | `1.78` | `1.88` | `1.81` | `4.31s` | `$0.009898` |

The downstream qualitative analysis uses the already-kept `gpt-5-mini`, `k=8` chain.

## Qualitative Analysis

The qualitative section:

- reuses the final generator chain directly
- scores each example with retrieval and generator metrics
- assigns a compact `case_label`
- displays 4 representative examples, one per label when available

Current label summary from the saved notebook outputs:

| Case label | Count |
|---|---:|
| `Alternative valid evidence retrieved` | `81` |
| `Good retrieval and grounded answer` | `12` |
| `Mixed behavior` | `5` |
| `Relevant retrieval but incomplete answer` | `2` |

## Setup

Create an environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your API key in the shell or in `.env`:

```bash
OPENAI_API_KEY=...
```

Then start Jupyter and open the notebook:

```bash
jupyter notebook
```

## Notes

- The notebook is evaluation-heavy and expensive to rerun end to end.
- OpenAI access is required for embeddings, answer generation, and LLM-as-judge scoring.
- The reranker step downloads `BAAI/bge-reranker-base` on first use.
- During the chunk-size-2000 experiment, two OpenAI API `400` messages were printed, but only one question failed at the outer evaluation-loop level. One inner judge error fell back to relevance score `0`, while one full question was skipped, leaving `99` valid evaluations.
- The qualitative analysis also shows one inner retriever-judge `400` error; it was handled locally and the analysis continued.
- Results are not fully deterministic because both generation and judging use LLMs.
