from django.contrib import admin
from .models import VectorEmbedding, VectorIndex, VectorSearchQuery, VectorSimilarity

@admin.register(VectorEmbedding)
class VectorEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('name', 'entity_type', 'embedding_model', 'embedding_dimension', 'project', 'created_at')
    list_filter = ('entity_type', 'embedding_model', 'project')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(VectorIndex)
class VectorIndexAdmin(admin.ModelAdmin):
    list_display = ('name', 'index_type', 'dimension', 'num_vectors', 'project', 'created_at')
    list_filter = ('index_type', 'project')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at', 'dimension', 'num_vectors')

@admin.register(VectorSearchQuery)
class VectorSearchQueryAdmin(admin.ModelAdmin):
    list_display = ('query_text', 'num_results', 'duration_ms', 'project', 'created_at')
    list_filter = ('project',)
    search_fields = ('query_text',)
    readonly_fields = ('created_at', 'duration_ms')

@admin.register(VectorSimilarity)
class VectorSimilarityAdmin(admin.ModelAdmin):
    list_display = ('source_embedding', 'target_embedding', 'similarity_score', 'created_at')
    list_filter = ('source_embedding__project', 'similarity_score')
    readonly_fields = ('created_at',)
