from django.db import models
from api.models import Project, Dataset, Model

class VectorEmbedding(models.Model):
    """Model for storing vector embeddings of various entities"""
    ENTITY_TYPES = [
        ('text', 'Text'),
        ('image', 'Image'),
        ('audio', 'Audio'),
        ('dataset', 'Dataset'),
        ('model', 'Model'),
        ('evaluation', 'Evaluation')
    ]
    
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    entity_type = models.CharField(max_length=20, choices=ENTITY_TYPES)
    embedding = models.BinaryField()  # Store serialized vector data
    embedding_model = models.CharField(max_length=100)  # Name of model used to generate embedding
    embedding_dimension = models.IntegerField()  # Dimensionality of the embedding
    metadata = models.JSONField(default=dict)  # Store additional metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='embeddings')
    
    # Optional relations
    dataset = models.ForeignKey(Dataset, on_delete=models.SET_NULL, null=True, blank=True, related_name='embeddings')
    model = models.ForeignKey(Model, on_delete=models.SET_NULL, null=True, blank=True, related_name='embeddings')
    
    class Meta:
        indexes = [
            models.Index(fields=['entity_type']),
            models.Index(fields=['project']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.entity_type})"


class VectorIndex(models.Model):
    """Model for storing vector index information"""
    INDEX_TYPES = [
        ('faiss', 'FAISS'),
        ('annoy', 'Annoy'),
        ('hnsw', 'HNSW'),
        ('pq', 'Product Quantization'),
        ('ivf', 'Inverted File'),
        ('flat', 'Flat Index')
    ]
    
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    index_type = models.CharField(max_length=20, choices=INDEX_TYPES)
    index_file = models.CharField(max_length=500)  # Path to the index file
    dimension = models.IntegerField()  # Dimensionality of vectors in this index
    num_vectors = models.IntegerField(default=0)  # Number of vectors in this index
    metadata = models.JSONField(default=dict)  # Store additional metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='vector_indices')
    
    class Meta:
        verbose_name_plural = "Vector Indices"
    
    def __str__(self):
        return f"{self.name} ({self.index_type})"


class VectorSearchQuery(models.Model):
    """Model for tracking vector search queries"""
    query_text = models.TextField()
    embedding = models.BinaryField(null=True, blank=True)
    num_results = models.IntegerField()
    filters = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    duration_ms = models.FloatField(null=True, blank=True)  # Query duration in milliseconds
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='vector_queries')
    
    class Meta:
        verbose_name_plural = "Vector Search Queries"
    
    def __str__(self):
        return f"Query: {self.query_text[:50]}..."


class VectorSimilarity(models.Model):
    """Model for storing similarity relationships between vectors"""
    source_embedding = models.ForeignKey(VectorEmbedding, on_delete=models.CASCADE, related_name='outgoing_similarities')
    target_embedding = models.ForeignKey(VectorEmbedding, on_delete=models.CASCADE, related_name='incoming_similarities')
    similarity_score = models.FloatField()  # Cosine similarity or other similarity measure
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Vector Similarities"
        unique_together = ('source_embedding', 'target_embedding')
        indexes = [
            models.Index(fields=['source_embedding', 'similarity_score']),
            models.Index(fields=['target_embedding', 'similarity_score']),
        ]
    
    def __str__(self):
        return f"{self.source_embedding.name} → {self.target_embedding.name} ({self.similarity_score:.3f})"
