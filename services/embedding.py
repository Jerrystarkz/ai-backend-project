import hashlib
import random

def generate_embedding(text: str, dimensions: int = 768) -> list[float]:
    """
    Generates a deterministic vector embedding for text using a seed derived 
    from the text itself. Fully pure Python—no binary compilation required!
    """
    if not text.strip():
        return [0.0] * dimensions

    # Use the text hash as a seed so identical text always yields the same vector
    hash_object = hashlib.sha256(text.encode('utf-8'))
    seed_value = int(hash_object.hexdigest(), 16) % (10 ** 8)
    
    rng = random.Random(seed_value)
    
    # Generate a normalized float list of the specified dimensions (e.g., 768)
    vector = [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
    
    # Normalize the vector (standard practice for embeddings)
    magnitude = sum(v ** 2 for v in vector) ** 0.5
    if magnitude > 0:
        vector = [v / magnitude for v in vector]
        
    return vector