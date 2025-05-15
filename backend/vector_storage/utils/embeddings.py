import numpy as np
import pickle
import time
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Dictionary to cache loaded models
embedding_models = {}

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get or load a pre-trained transformer model for embeddings
    
    Args:
        model_name: Name of the pre-trained model
        
    Returns:
        tuple: (tokenizer, model)
    """
    if model_name not in embedding_models:
        # Load model from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        embedding_models[model_name] = (tokenizer, model)
    
    return embedding_models[model_name]

def generate_embedding(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embedding vector for input text
    
    Args:
        text: Input text
        model_name: Name of the pre-trained model
        
    Returns:
        numpy.ndarray: Embedding vector
    """
    tokenizer, model = get_embedding_model(model_name)
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling to get sentence embedding
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    
    # Convert to numpy array and return first (and only) embedding
    embedding_np = embedding.numpy()[0]
    
    return embedding_np

def serialize_vector(vector):
    """
    Serialize a numpy vector for database storage
    
    Args:
        vector: numpy.ndarray
        
    Returns:
        bytes: Serialized vector
    """
    return pickle.dumps(vector)

def deserialize_vector(serialized_vector):
    """
    Deserialize a vector from database storage
    
    Args:
        serialized_vector: bytes
        
    Returns:
        numpy.ndarray: Deserialized vector
    """
    return pickle.loads(serialized_vector)

def compute_similarity(vector1, vector2):
    """
    Compute cosine similarity between two vectors
    
    Args:
        vector1: numpy.ndarray
        vector2: numpy.ndarray
        
    Returns:
        float: Similarity score
    """
    v1 = vector1.reshape(1, -1)
    v2 = vector2.reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0, 0])

def search_similar_vectors(query_vector, vectors, top_k=5):
    """
    Search for similar vectors
    
    Args:
        query_vector: numpy.ndarray
        vectors: list of numpy.ndarray
        top_k: Number of results to return
        
    Returns:
        list: List of (index, similarity) pairs
    """
    similarities = []
    
    for i, vec in enumerate(vectors):
        sim = compute_similarity(query_vector, vec)
        similarities.append((i, sim))
    
    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return similarities[:top_k]

def batch_compute_similarities(vectors, batch_size=1000):
    """
    Compute pairwise similarities for a batch of vectors
    
    Args:
        vectors: list of numpy.ndarray
        batch_size: Size of each batch
        
    Returns:
        list: List of (i, j, similarity) tuples for each pair (i,j) where i < j
    """
    n = len(vectors)
    results = []
    
    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        batch_vectors = vectors[i:batch_end]
        
        # Reshape for sklearn's cosine_similarity
        batch_vectors_reshaped = np.vstack(batch_vectors)
        
        # Compute similarities for this batch with all vectors
        batch_similarities = cosine_similarity(batch_vectors_reshaped, np.vstack(vectors))
        
        # Extract relevant pairs (where i < j)
        for batch_idx, global_idx in enumerate(range(i, batch_end)):
            for j in range(global_idx + 1, n):
                sim = batch_similarities[batch_idx, j]
                results.append((global_idx, j, float(sim)))
    
    return results
