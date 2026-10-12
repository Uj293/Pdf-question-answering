# Research RAG: a retrieval experiment over a research paper

This project is a small retrieval-augmented generation (RAG) system built around the paper [*Attention Is All You Need*](data/papers/attention_is_all_you_need.pdf). It extracts text from the PDF, splits it into chunks, embeds those chunks, retrieves the most relevant context for a question, and asks a local Ollama model to produce an answer grounded in that context.

The repository also contains a small, explicit retrieval evaluation. The goal is not to claim a generally accurate RAG system from five questions; it is to make each retrieval change observable and reproducible.

## What the application does

When you run the project, it performs these steps:

1. Reads the paper from `data/papers/attention_is_all_you_need.pdf` with PyMuPDF.
2. Splits the extracted text into overlapping chunks of 1,000 characters with a 150-character overlap.
3. Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
4. Retrieves the top three chunks for a demonstration question with hybrid retrieval.
5. Sends the question and retrieved context to the local Ollama model `qwen2.5:3b`.
6. Prints the generated answer. The prompt instructs Ollama to cite supplied source numbers.
7. Runs Hit Rate@3 over the five labelled evaluation questions.

The current demonstration question is:

```text
Why does the Transformer use positional encoding?
```

Change `DEMO_QUESTION` in `src/research_rag/main.py` to ask another question about the paper.

## Architecture

```text
PDF
  -> text extraction (PyMuPDF)
  -> overlapping chunks
  -> dense embeddings (all-MiniLM-L6-v2)
  -> hybrid retrieval
       ├─ cosine similarity over embeddings
       └─ BM25 lexical ranking
  -> top 3 chunks
  -> Ollama / qwen2.5:3b
  -> grounded answer with source references
```

### Hybrid retrieval

Dense embeddings are useful when a question and a passage use different wording but mean the same thing. They can, however, underweight a specific technical phrase. The positional-encoding question exposed that limitation: the dense retriever preferred broadly Transformer-related chunks instead of the section explaining sequence order.

`src/research_rag/retrieval.py` combines two signals after min-max normalization:

```text
final score = 0.65 × semantic cosine score + 0.35 × BM25 lexical score
```

BM25 increases the influence of terms such as `positional` and `encoding`; cosine similarity retains semantic matching. The weights are experiment settings, not universal constants.

## Setup

### Prerequisites

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)

Install the Python dependencies:

```bash
uv sync
```

Start Ollama in a separate terminal if it is not already running:

```bash
ollama serve
```

Download the model used by `src/research_rag/generation.py`:

```bash
ollama pull qwen2.5:3b
```

Run the application:

```bash
uv run python -m research_rag.main
```

The embedding model is downloaded from Hugging Face on its first run. An unauthenticated Hugging Face warning affects rate limits only; it does not change the retrieval logic.

## Experiment history

### 1. Dense-retrieval baseline

The initial system used cosine similarity only. With the original chunking configuration, the five-question evaluation recorded a Hit Rate@3 of 80%.

Hit Rate@k counts a question as a hit when at least one manually labelled relevant chunk appears in the first `k` retrieved results:

```text
Hit@k = questions with at least one relevant top-k chunk / total questions
```

### 2. Smaller-chunk experiment

The chunk configuration was changed to:

```python
chunk_size=1000
overlap=150
```

This produced 41 chunks. The first evaluation run showed 20%, but that result was invalid: the relevant chunk IDs still belonged to the previous chunk layout. A chunk ID identifies a position in a particular chunking run, not a permanent passage in the document.

### 3. Relabel the evaluation set

Each question was relabelled against the new 41-chunk layout. The current labels in `main.py` are:

| Question | Relevant chunk IDs |
| --- | --- |
| Transformer architecture | 7, 8, 9 |
| Multi-head attention | 12, 13, 14 |
| Positional encoding | 17, 18, 19 |
| Advantages of self-attention | 20, 21, 22, 23 |
| Machine-translation evaluation | 24, 26, 27, 28, 29 |

After relabelling, dense retrieval returned to 80% Hit Rate@3. The unresolved question was positional encoding: the relevant chunks 17–19 were not in its top three.

### 4. Hybrid retrieval

Hybrid retrieval was added to preserve semantic matching while promoting exact research-paper terminology. The same labelled five-question evaluation then produced:

```text
Hit Rate@3: 100.00% (5/5)
```

For the positional-encoding question, the retrieved chunk IDs became `[17, 18, 26]`, so two directly relevant chunks now appear in the top three.

This is a useful regression check for this one paper and five questions, not proof that the system will score 100% on unseen documents or broader questions.

## Project layout

```text
data/papers/attention_is_all_you_need.pdf  Source document
src/research_rag/chunking.py               Overlapping character-based chunker
src/research_rag/embeddings.py             SentenceTransformer embedding model
src/research_rag/retrieval.py              Cosine, BM25, and hybrid ranking
src/research_rag/generation.py             Ollama prompt and answer generation
src/research_rag/evaluation.py             Hit Rate@k calculation
src/research_rag/main.py                   End-to-end application and evaluation
```

## Evaluation output

The program prints, for every evaluation question:

```text
Question: ...
Relevant chunks: [...]
Retrieved chunks: [...]
Hit@3: True | False
```

At the end, it prints the aggregate result. Inspect both the aggregate metric and the per-question chunk IDs: a high score can still hide a weak question or weak labels.

## Limitations and next steps

- The document is small and the evaluation set has only five questions.
- Chunk IDs are tied to the exact document extraction and chunk configuration. Rebuild labels whenever either changes.
- The BM25 implementation is intentionally lightweight and recalculates its index for each query. Persist an index for larger document collections.
- Test alternative embedding models and hybrid-score weights on a held-out evaluation set rather than tuning to these same five questions.
- Add Mean Reciprocal Rank (MRR), precision@k, and answer-faithfulness evaluation.
- Store chunk metadata such as page number and section title, then cite that metadata in generated answers.
- Add a command-line interface so users can ask arbitrary questions instead of editing `DEMO_QUESTION`.
