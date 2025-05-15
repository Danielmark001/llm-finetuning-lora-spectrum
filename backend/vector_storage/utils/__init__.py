# Package initialization file
from .embeddings import (
    generate_embedding,
    serialize_vector,
    deserialize_vector,
    compute_similarity,
    search_similar_vectors,
    batch_compute_similarities
)

from .vector_index import (
    VectorIndexManager,
    InMemoryVectorStore
)

from .vector_operations import (
    create_embedding_for_text,
    create_index_for_embeddings,
    search_vectors
)

__all__ = [
    'generate_embedding',
    'serialize_vector',
    'deserialize_vector',
    'compute_similarity',
    'search_similar_vectors',
    'batch_compute_similarities',
    'VectorIndexManager',
    'InMemoryVectorStore',
    'create_embedding_for_text',
    'create_index_for_embeddings',
    'search_vectors'
]
