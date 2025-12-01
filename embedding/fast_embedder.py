from sentence_transformers import SentenceTransformer
import numpy as np

class FastCodeEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code')
        self.model.eval()
        
    def embed(self, code: str) -> np.ndarray:
        emb = self.model.encode(code, convert_to_numpy=True, normalize_embeddings=True)
        return emb