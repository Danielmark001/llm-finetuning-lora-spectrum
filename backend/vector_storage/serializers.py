from rest_framework import serializers
from .models import VectorEmbedding, VectorIndex, VectorSearchQuery, VectorSimilarity
import base64
import numpy as np
from .utils import deserialize_vector

class VectorEmbeddingSerializer(serializers.ModelSerializer):
    embedding_data = serializers.SerializerMethodField()
    
    class Meta:
        model = VectorEmbedding
        fields = [
            'id', 'name', 'description', 'entity_type', 'embedding_model',
            'embedding_dimension', 'metadata', 'created_at', 'updated_at',
            'project', 'dataset', 'model', 'embedding_data'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'embedding_data']
    
    def get_embedding_data(self, obj):
        """Return a base64 representation of the embedding for API responses"""
        try:
            # Deserialize the embedding
            embedding = deserialize_vector(obj.embedding)
            
            # Convert to list and then to base64 for transmission
            embedding_list = embedding.tolist()
            return {
                'dimension': obj.embedding_dimension,
                'preview': embedding_list[:10],  # First 10 dimensions as preview
                'model': obj.embedding_model
            }
        except Exception as e:
            return {
                'error': str(e),
                'dimension': obj.embedding_dimension,
                'model': obj.embedding_model
            }

class VectorEmbeddingCreateSerializer(serializers.ModelSerializer):
    content = serializers.CharField(write_only=True, required=True)
    
    class Meta:
        model = VectorEmbedding
        fields = [
            'name', 'description', 'entity_type', 'content', 
            'metadata', 'project', 'dataset', 'model'
        ]
    
    def validate(self, attrs):
        # Remove content from attrs as it's not a model field
        content = attrs.pop('content')
        # Add it back for use in create
        attrs['content'] = content
        return attrs

class VectorIndexSerializer(serializers.ModelSerializer):
    class Meta:
        model = VectorIndex
        fields = [
            'id', 'name', 'description', 'index_type', 'dimension',
            'num_vectors', 'metadata', 'created_at', 'updated_at', 'project'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'dimension', 'num_vectors']

class VectorIndexCreateSerializer(serializers.ModelSerializer):
    embedding_ids = serializers.ListField(
        child=serializers.IntegerField(),
        write_only=True,
        required=True
    )
    
    class Meta:
        model = VectorIndex
        fields = [
            'name', 'description', 'index_type', 'project', 'embedding_ids'
        ]
    
    def validate(self, attrs):
        # Remove embedding_ids from attrs as it's not a model field
        embedding_ids = attrs.pop('embedding_ids')
        # Add it back for use in create
        attrs['embedding_ids'] = embedding_ids
        return attrs

class VectorSearchQuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = VectorSearchQuery
        fields = [
            'id', 'query_text', 'num_results', 'filters',
            'created_at', 'duration_ms', 'project'
        ]
        read_only_fields = ['id', 'created_at', 'duration_ms']

class VectorSimilaritySerializer(serializers.ModelSerializer):
    source_name = serializers.SerializerMethodField()
    target_name = serializers.SerializerMethodField()
    
    class Meta:
        model = VectorSimilarity
        fields = [
            'id', 'source_embedding', 'target_embedding', 
            'source_name', 'target_name',
            'similarity_score', 'created_at'
        ]
        read_only_fields = ['id', 'created_at', 'source_name', 'target_name']
    
    def get_source_name(self, obj):
        return obj.source_embedding.name
    
    def get_target_name(self, obj):
        return obj.target_embedding.name

class VectorSearchRequestSerializer(serializers.Serializer):
    query = serializers.CharField(required=True)
    index_id = serializers.IntegerField(required=True)
    top_k = serializers.IntegerField(required=False, default=10)
    filters = serializers.DictField(required=False, default=dict)

class VectorSearchResultSerializer(serializers.Serializer):
    id = serializers.IntegerField(source='embedding.id')
    name = serializers.CharField(source='embedding.name')
    description = serializers.CharField(source='embedding.description')
    entity_type = serializers.CharField(source='embedding.entity_type')
    metadata = serializers.JSONField(source='embedding.metadata')
    distance = serializers.FloatField()
    score = serializers.FloatField()
    
    class Meta:
        fields = [
            'id', 'name', 'description', 'entity_type', 
            'metadata', 'distance', 'score'
        ]
