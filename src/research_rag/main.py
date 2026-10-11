
import pymupdf

from research_rag.chunking import chunk_text
from research_rag.embeddings import create_embeddings, create_query_embedding
from research_rag.retrieval import retrieve_top_k
from research_rag.evaluation import hit_rate_at_k


def main() -> None:
    pdf_path = "data/papers/attention_is_all_you_need.pdf"

    # 1. Extract text from the research paper.
    with pymupdf.open(pdf_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    # 2. Split the extracted text into chunks.
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        if "positional encoding" in chunk.lower() or "sequence order" in chunk.lower():
            print(f"\n{'=' * 15} CHUNK {index} {'=' * 15}")
            print(chunk)


    # Debug: inspect chunks related to positional encoding.
    for index in [4, 9, 10, 11, 12, 14]:
        print(f"\n{'=' * 25} CHUNK {index} {'=' * 25}")
        print(chunks[index])


    # 3. Generate embeddings for all chunks once.
    embeddings = create_embeddings(chunks)

    # 4. Define evaluation questions and relevant chunk IDs.
    # These labels are provisional and should be verified by
    # inspecting the full contents of the relevant chunks.
    evaluation_questions = [
        {
            "question": "What is the Transformer architecture?",
            "relevant_chunks": [4],
        },
        {
            "question": "How does multi-head attention work?",
            "relevant_chunks": [5, 6],
        },
        {
            "question": "Why does the Transformer use positional encoding?",
            "relevant_chunks": [9, 10, 11],
        },
        {
            "question": (
                "What are the advantages of self-attention "
                "over recurrent layers?"
            ),
            "relevant_chunks": [9, 10],
        },
        {
            "question": (
                "How was the Transformer evaluated "
                "on machine translation?"
            ),
            "relevant_chunks": [12, 13, 14],
        },
    ]

    # 5. Evaluate retrieval performance at k=3.
    hit_rate_at_k(
        questions=evaluation_questions,
        chunks=chunks,
        embeddings=embeddings,
        create_query_embedding=create_query_embedding,
        retrieve_top_k=retrieve_top_k,
        k=3,
    )


if __name__ == "__main__":
    main()
