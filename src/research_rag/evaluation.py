
def hit_rate_at_k(
    questions: list[dict],
    chunks: list[str],
    embeddings,
    create_query_embedding,
    retrieve_top_k,
    k: int = 3,
) -> float:
    if not questions:
        raise ValueError("questions cannot be empty")

    hits = 0

    for item in questions:
        query = item["question"]
        relevant_chunks = set(item["relevant_chunks"])

        query_embedding = create_query_embedding(query)
        results = retrieve_top_k(
            query_embedding,
            embeddings,
            chunks,
            k=k,
        )

        retrieved_chunks = {
            result["chunk_index"] for result in results
        }

        hit = bool(retrieved_chunks & relevant_chunks)
        hits += int(hit)

        print(f"\nQuestion: {query}")
        print(f"Relevant chunks: {sorted(relevant_chunks)}")
        print(f"Retrieved chunks: {sorted(retrieved_chunks)}")
        print(f"Hit@{k}: {hit}")

    score = hits / len(questions)
    print(f"\nHit Rate@{k}: {score:.2%} ({hits}/{len(questions)})")

    return score
