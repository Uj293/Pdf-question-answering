import numpy as np


def cosine_similarity(query_embedding, chunk_embeddings):
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)

    similarities = np.dot(chunk_embeddings, query_embedding) / (
        chunk_norms * query_norm
    )

    return similarities



def retrieve_top_k(query_embedding, chunk_embeddings, chunks, k: int = 3):
    similarities = cosine_similarity(query_embedding, chunk_embeddings)
    top_indices = np.argsort(similarities)[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "chunk_index": int(index),
            "chunk": chunks[index],
            "score": float(similarities[index]),
        })

    return results
