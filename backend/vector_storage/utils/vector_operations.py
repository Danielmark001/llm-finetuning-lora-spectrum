from django.db import models
from django.apps import apps
import os
import numpy as np
import pickle
import json
from .embeddings import generate_embedding, serialize_vector, deserialize_vector
from .vector_index import VectorIndexManager
import time

def create_embedding_for_text(text, project_id, name=None, description=None, model_name="sentence-transformers/all-MiniLM-L6-v2", entity_type="text", metadata=None, **kwargs):
    """
    Create a vector embedding for a text and save it to the database
    
    Args:
        text: Input text
        project_id: Project ID
        name: Name for the embedding (defaults to truncated text)
        description: Description
        model_name: Name of the embedding model
        entity_type: Type of entity
        metadata: Additional metadata
        **kwargs: Additional fields for the VectorEmbedding model
        
    Returns:
        VectorEmbedding: Created embedding object
    """
    # Import here to avoid circular imports
    VectorEmbedding = apps.get_model('vector_storage', 'VectorEmbedding')
    
    # Generate embedding
    embedding_vector = generate_embedding(text, model_name)
    serialized_embedding = serialize_vector(embedding_vector)
    
    # Prepare name if not provided
    if name is None:
        name = text[:50] + ('...' if len(text) > 50 else '')
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    metadata['text'] = text
    
    # Create and save the embedding
    embedding = VectorEmbedding(
        name=name,
        description=description or '',
        entity_type=entity_type,
        embedding=serialized_embedding,
        embedding_model=model_name,
        embedding_dimension=embedding_vector.shape[0],
        metadata=metadata,
        project_id=project_id,
        **kwargs
    )
    embedding.save()
    
    return embedding

def create_index_for_embeddings(embeddings, index_type="flat", index_params=None, name=None, description=None):
    """
    Create a vector index for a set of embeddings
    
    Args:
        embeddings: List of VectorEmbedding objects
        index_type: Type of index
        index_params: Parameters for the index
        name: Name for the index
        description: Description
        
    Returns:
        VectorIndex: Created index object
    """
    # Import here to avoid circular imports
    VectorIndex = apps.get_model('vector_storage', 'VectorIndex')
    
    if not embeddings:
        raise ValueError("No embeddings provided")
    
    # Get project ID from the first embedding
    project_id = embeddings[0].project_id
    
    # Extract vectors
    vectors = [deserialize_vector(emb.embedding) for emb in embeddings]
    vectors_array = np.array(vectors).astype('float32')
    
    # Create directory for indices if it doesn't exist
    indices_dir = os.path.join('media', 'vector_indices')
    os.makedirs(indices_dir, exist_ok=True)
    
    # Generate a unique filename for the index
    timestamp = int(time.time())
    index_filename = f"index_{project_id}_{timestamp}.idx"
    index_path = os.path.join(indices_dir, index_filename)
    
    # Create the index
    _, metadata = VectorIndexManager.create_index(
        vectors_array, 
        index_type=index_type, 
        index_params=index_params,
        index_path=index_path
    )
    
    # Prepare the index object
    dimension = metadata['dimension']
    num_vectors = metadata['num_vectors']
    
    # Default name if not provided
    if name is None:
        name = f"{index_type.upper()} Index - {num_vectors} vectors"
    
    # Create and save the index object
    index = VectorIndex(
        name=name,
        description=description or '',
        index_type=index_type,
        index_file=index_path,
        dimension=dimension,
        num_vectors=num_vectors,
        metadata={
            'parameters': index_params or {},
            'embedding_ids': [emb.id for emb in embeddings],
            'embedding_model': embeddings[0].embedding_model,
        },
        project_id=project_id
    )
    index.save()
    
    return index

def search_vectors(query_text, index_id, top_k=10, model_name=None):
    """
    Search for similar vectors using a text query
    
    Args:
        query_text: Text query
        index_id: ID of the VectorIndex to search
        top_k: Number of results to return
        model_name: Name of the embedding model (if None, use the one from the index)
        
    Returns:
        dict: Search results with embedding objects and scores
    """
    # Import here to avoid circular imports
    VectorIndex = apps.get_model('vector_storage', 'VectorIndex')
    VectorEmbedding = apps.get_model('vector_storage', 'VectorEmbedding')
    VectorSearchQuery = apps.get_model('vector_storage', 'VectorSearchQuery')
    
    # Get the index
    try:
        index_obj = VectorIndex.objects.get(id=index_id)
    except VectorIndex.DoesNotExist:
        raise ValueError(f"Index with ID {index_id} does not exist")
    
    # Load the model name from the index if not provided
    if model_name is None:
        model_name = index_obj.metadata.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2")
    
    # Record the start time
    start_time = time.time()
    
    # Generate embedding for the query
    query_vector = generate_embedding(query_text, model_name)
    serialized_query = serialize_vector(query_vector)
    
    # Load the index
    faiss_index = VectorIndexManager.load_index(index_obj.index_file)
    
    # Search the index
    distances, indices = VectorIndexManager.search_index(faiss_index, query_vector, top_k)
    
    # Record the end time
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    
    # Get the embedding IDs from the index metadata
    embedding_ids = index_obj.metadata.get('embedding_ids', [])
    
    # Prepare results
    results = []
    for i, idx in enumerate(indices):
        if idx < len(embedding_ids):
            embedding_id = embedding_ids[idx]
            try:
                embedding = VectorEmbedding.objects.get(id=embedding_id)
                results.append({
                    'embedding': embedding,
                    'distance': float(distances[i]),
                    'score': 1.0 - float(distances[i]) / 2.0,  # Convert L2 distance to similarity score
                })
            except VectorEmbedding.DoesNotExist:
                # Skip embeddings that no longer exist
                continue
    
    # Record the search query
    VectorSearchQuery.objects.create(
        query_text=query_text,
        embedding=serialized_query,
        num_results=len(results),
        filters={},
        duration_ms=duration_ms,
        project_id=index_obj.project_id
    )
    
    return {
        'results': results,
        'query_time_ms': duration_ms,
        'total_results': len(results),
    }
