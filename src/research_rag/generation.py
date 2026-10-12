
# import os

# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

# client = OpenAI(api_key=api_key)


# def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
#     context = "\n\n".join(
#         f"[Source {i + 1}]\n{result['chunk']}"
#         for i, result in enumerate(retrieved_chunks)
#     )

#     response = client.responses.create(
#         model="gpt-4.1-mini",
#         instructions=(
#             "You are a research paper assistant. Answer using only the "
#             "provided context. If the context is insufficient, say so. "
#             "Do not invent facts. Refer to the supplied source numbers "
#             "when supporting your answer."
#         ),
#         input=f"""
# Question:
# {question}

# Research paper context:
# {context}
# """,
#     )

#     return response.output_text



import ollama


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source {i + 1}]\n{result['chunk']}"
        for i, result in enumerate(retrieved_chunks)
    )

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research paper assistant. Answer using only "
                    "the provided context. If the context is insufficient, "
                    "say so. Do not invent facts. Every factual paragraph "
                    "must end with one or more citations in the form "
                    "[Source 1]. Do not cite a source that does not support "
                    "the statement."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Research paper context:\n{context}"
                ),
            },
        ],
    )

    return response["message"]["content"]

