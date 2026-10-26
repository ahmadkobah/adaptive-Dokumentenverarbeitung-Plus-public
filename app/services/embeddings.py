"""
Embeddings Service — Plus Tier
SentenceTransformer for semantic search vector index.
DeepSeek handles everything else; this is the only local model.
"""

import logging
from typing import List

logger = logging.getLogger("embeddings")

_model = None
_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model ready")
    return _model


def embed(text: str) -> List[float]:
    """Return a 384-dim embedding vector for the given text."""
    if not text or not text.strip():
        return []
    model = get_model()
    vector = model.encode(text, convert_to_tensor=False)
    return vector.tolist()
