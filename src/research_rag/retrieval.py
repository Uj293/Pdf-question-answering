from collections import Counter
import math
import re

import numpy as np


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "does", "for", "from",
    "how", "in", "is", "of", "on", "or", "the", "to", "use", "was",
    "what", "why", "with",
}


def tokenize(text: str) -> list[str]:
    """Return normalized content words for lexical retrieval."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word.rstrip("s") for word in words if word not in STOP_WORDS]


def cosine_similarity(query_embedding, chunk_embeddings):
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)

    similarities = np.dot(chunk_embeddings, query_embedding) / (
        chunk_norms * query_norm
    )

    return similarities


def bm25_scores(query: str, chunks: list[str]) -> np.ndarray:
    """Score chunks using BM25 so exact technical terms influence ranking."""
    query_terms = tokenize(query)
    documents = [tokenize(chunk) for chunk in chunks]

    if not query_terms or not documents:
        return np.zeros(len(chunks))

    document_frequencies = Counter(
        term for document in documents for term in set(document)
    )
    average_length = sum(len(document) for document in documents) / len(documents)
    k1, b = 1.5, 0.75
    scores = np.zeros(len(chunks))

    for index, document in enumerate(documents):
        term_frequencies = Counter(document)
        for term in query_terms:
            frequency = term_frequencies[term]
            if not frequency:
                continue

            inverse_document_frequency = math.log(
                1 + (len(documents) - document_frequencies[term] + 0.5)
                / (document_frequencies[term] + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * len(document) / average_length
            )
            scores[index] += inverse_document_frequency * frequency * (k1 + 1) / denominator

    return scores


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    minimum, maximum = scores.min(), scores.max()
    if maximum == minimum:
        return np.zeros_like(scores)
    return (scores - minimum) / (maximum - minimum)


def retrieve_top_k(
    query_embedding,
    chunk_embeddings,
    chunks: list[str],
    k: int = 3,
    query: str | None = None,
):
    """Retrieve chunks with semantic search, optionally blended with BM25."""
    semantic_scores = cosine_similarity(query_embedding, chunk_embeddings)
    lexical_scores = bm25_scores(query, chunks) if query else np.zeros(len(chunks))
    combined_scores = (
        0.65 * min_max_normalize(semantic_scores)
        + 0.35 * min_max_normalize(lexical_scores)
        if query
        else semantic_scores
    )
    top_indices = np.argsort(combined_scores)[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "chunk_index": int(index),
            "chunk": chunks[index],
            "score": float(combined_scores[index]),
            "semantic_score": float(semantic_scores[index]),
            "lexical_score": float(lexical_scores[index]),
        })

    return results
