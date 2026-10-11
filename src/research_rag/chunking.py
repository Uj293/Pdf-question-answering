
def chunk_text(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Split unusually long paragraphs into smaller pieces.
        while len(paragraph) > chunk_size:
            piece = paragraph[:chunk_size]
            paragraph = paragraph[chunk_size:]

            if current_chunk:
                chunks.append(current_chunk)

            chunks.append(piece)
            current_chunk = ""

        candidate = (
            f"{current_chunk}\n{paragraph}"
            if current_chunk
            else paragraph
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # Preserve some context across chunk boundaries.
            current_chunk = (
                current_chunk[-overlap:] + "\n" + paragraph
                if overlap and current_chunk
                else paragraph
            )

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
