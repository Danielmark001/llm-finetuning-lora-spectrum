import os
import json
import numpy as np
import faiss
import pickle
import time

class VectorIndexManager:
    """Manager for vector indices operations"""
    
    @staticmethod
    def create_index(vectors, index_type="flat", index_params=None, index_path=None):
        """
        Create a vector index using FAISS
        
        Args:
            vectors: List of numpy vectors or numpy array with shape (n, dim)
            index_type: Type of index ('flat', 'ivf', 'pq', 'hnsw')
            index_params: Parameters for the index
            index_path: Path to save the index
            
        Returns:
            tuple: (index, metadata)
        """
        if isinstance(vectors, list):
            vectors = np.array(vectors).astype('float32')
        
        n, dim = vectors.shape
        
        # Set default parameters if none provided
        if index_params is None:
            index_params = {}
        
        # Create the appropriate index based on type
        if index_type == "flat":
            index = faiss.IndexFlatL2(dim)
        elif index_type == "ivf":
            # IVF index requires clustering, use a flat index for now
            nlist = index_params.get("nlist", min(int(np.sqrt(n)), 100))
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            # IVF indices need to be trained
            index.train(vectors)
        elif index_type == "pq":
            # Product Quantization
            m = index_params.get("m", min(dim // 2, 8))  # Number of subquantizers
            nbits = index_params.get("nbits", 8)
            index = faiss.IndexPQ(dim, m, nbits)
            index.train(vectors)
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World
            M = index_params.get("M", 16)  # Number of connections per layer
            index = faiss.IndexHNSWFlat(dim, M)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add vectors to the index
        index.add(vectors)
        
        # Save index if path provided
        if index_path:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(index, index_path)
        
        # Prepare metadata
        metadata = {
            "index_type": index_type,
            "dimension": dim,
            "num_vectors": n,
            "parameters": index_params,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return index, metadata
    
    @staticmethod
    def load_index(index_path):
        """
        Load a saved FAISS index
        
        Args:
            index_path: Path to the index file
            
        Returns:
            faiss.Index: Loaded index
        """
        return faiss.read_index(index_path)
    
    @staticmethod
    def search_index(index, query_vector, k=10):
        """
        Search for nearest vectors in the index
        
        Args:
            index: FAISS index
            query_vector: Query vector (numpy array)
            k: Number of results to return
            
        Returns:
            tuple: (distances, indices)
        """
        # Ensure query_vector is in the right shape and type
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        
        # Search the index
        distances, indices = index.search(query_vector, k)
        
        return distances[0], indices[0]
    
    @staticmethod
    def batch_search_index(index, query_vectors, k=10):
        """
        Search for nearest vectors for multiple queries
        
        Args:
            index: FAISS index
            query_vectors: Query vectors (numpy array with shape (n_queries, dim))
            k: Number of results to return per query
            
        Returns:
            tuple: (distances, indices)
        """
        # Ensure query_vectors is in the right shape and type
        query_vectors = np.array(query_vectors).astype('float32')
        
        # Search the index
        distances, indices = index.search(query_vectors, k)
        
        return distances, indices


class InMemoryVectorStore:
    """Simple in-memory vector store for development and testing"""
    
    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.index = None
    
    def add_vector(self, vector, metadata=None):
        """Add a vector with optional metadata"""
        self.vectors.append(vector)
        self.metadata.append(metadata or {})
        # Invalidate index
        self.index = None
        return len(self.vectors) - 1
    
    def add_vectors(self, vectors, metadata_list=None):
        """Add multiple vectors with optional metadata"""
        if metadata_list is None:
            metadata_list = [{} for _ in vectors]
        
        for vector, metadata in zip(vectors, metadata_list):
            self.add_vector(vector, metadata)
        
        return list(range(len(self.vectors) - len(vectors), len(self.vectors)))
    
    def get_vector(self, index):
        """Get a vector by index"""
        return self.vectors[index]
    
    def get_metadata(self, index):
        """Get metadata for a vector"""
        return self.metadata[index]
    
    def build_index(self, index_type="flat"):
        """Build a search index"""
        vectors_array = np.array(self.vectors).astype('float32')
        self.index, _ = VectorIndexManager.create_index(vectors_array, index_type)
        return self.index
    
    def search(self, query_vector, k=10):
        """Search for similar vectors"""
        if self.index is None:
            self.build_index()
        
        distances, indices = VectorIndexManager.search_index(self.index, query_vector, k)
        
        results = []
        for i, idx in enumerate(indices):
            if idx < len(self.vectors):  # Ensure index is valid
                results.append({
                    "index": int(idx),
                    "distance": float(distances[i]),
                    "metadata": self.metadata[idx]
                })
        
        return results
    
    def save(self, path):
        """Save the vector store to disk"""
        data = {
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path):
        """Load a vector store from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        store = cls()
        store.vectors = data["vectors"]
        store.metadata = data["metadata"]
        
        return store
