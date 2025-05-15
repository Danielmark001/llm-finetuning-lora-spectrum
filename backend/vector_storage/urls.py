from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    VectorEmbeddingViewSet,
    VectorIndexViewSet,
    VectorSearchQueryViewSet,
    VectorSimilarityViewSet,
    VectorSearchAPIView
)

router = DefaultRouter()
router.register(r'embeddings', VectorEmbeddingViewSet)
router.register(r'indices', VectorIndexViewSet)
router.register(r'search-history', VectorSearchQueryViewSet)
router.register(r'similarities', VectorSimilarityViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('search/', VectorSearchAPIView.as_view(), name='vector-search'),
]
