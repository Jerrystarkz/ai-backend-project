def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Splits text into overlapping chunks of specified character length.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        # Move forward by chunk_size minus overlap
        start += chunk_size - overlap

    return chunks

import numpy as np

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))