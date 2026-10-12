
import pymupdf

from research_rag.chunking import chunk_text
from research_rag.embeddings import create_embeddings, create_query_embedding
from research_rag.evaluation import hit_rate_at_k
from research_rag.generation import generate_answer
from research_rag.retrieval import retrieve_top_k


DEMO_QUESTION = "Why does the Transformer use positional encoding?"


def main() -> None:
    pdf_path = "data/papers/attention_is_all_you_need.pdf"

    # 1. Extract text from the research paper.
    with pymupdf.open(pdf_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    # 2. Split the extracted text into chunks.
    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap=150,
    )
    print(f"Total chunks: {len(chunks)}")

    # 3. Generate embeddings for all chunks once.
    embeddings = create_embeddings(chunks)

    # 4. Retrieve context and generate a grounded answer with Ollama.
    demo_query_embedding = create_query_embedding(DEMO_QUESTION)
    retrieved_chunks = retrieve_top_k(
        demo_query_embedding,
        embeddings,
        chunks,
        k=3,
        query=DEMO_QUESTION,
    )
    answer = generate_answer(DEMO_QUESTION, retrieved_chunks)

    print(f"\nQuestion: {DEMO_QUESTION}")
    print("Retrieved chunk IDs:", [item["chunk_index"] for item in retrieved_chunks])
    print(f"\nAnswer:\n{answer}")

    # 5. Define evaluation questions and relevant chunk IDs.
    evaluation_questions = [
        {
            "question": "What is the Transformer architecture?",
            "relevant_chunks": [7, 8, 9],
        },
        {
            "question": "How does multi-head attention work?",
            "relevant_chunks": [12, 13, 14],
        },
        {
            "question": "Why does the Transformer use positional encoding?",
            "relevant_chunks": [17, 18, 19],
        },
        {
            "question": (
                "What are the advantages of self-attention "
                "over recurrent layers?"
            ),
            "relevant_chunks": [20, 21, 22, 23],
        },
        {
            "question": (
                "How was the Transformer evaluated "
                "on machine translation?"
            ),
            "relevant_chunks": [24, 26, 27, 28, 29],
        },
    ]

    # 6. Evaluate retrieval performance at k=3.
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
