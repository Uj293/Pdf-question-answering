from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(texts: list[str]):
    return model.encode(texts)


def create_query_embedding(query: str):
    return model.encode([query])[0]