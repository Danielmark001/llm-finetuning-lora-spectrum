from rest_framework import viewsets, status, generics
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import VectorEmbedding, VectorIndex, VectorSearchQuery, VectorSimilarity
from .serializers import (
    VectorEmbeddingSerializer, VectorEmbeddingCreateSerializer,
    VectorIndexSerializer, VectorIndexCreateSerializer,
    VectorSearchQuerySerializer, VectorSimilaritySerializer,
    VectorSearchRequestSerializer, VectorSearchResultSerializer
)
from .utils import (
    create_embedding_for_text, 
    create_index_for_embeddings,
    search_vectors
)

class VectorEmbeddingViewSet(viewsets.ModelViewSet):
    queryset = VectorEmbedding.objects.all()
    
    def get_serializer_class(self):
        if self.action == 'create':
            return VectorEmbeddingCreateSerializer
        return VectorEmbeddingSerializer
    
    def get_queryset(self):
        queryset = VectorEmbedding.objects.all()
        
        # Filter by project
        project_id = self.request.query_params.get('project_id')
        if project_id:
            queryset = queryset.filter(project_id=project_id)
        
        # Filter by entity type
        entity_type = self.request.query_params.get('entity_type')
        if entity_type:
            queryset = queryset.filter(entity_type=entity_type)
        
        # Filter by dataset
        dataset_id = self.request.query_params.get('dataset_id')
        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id)
        
        # Filter by model
        model_id = self.request.query_params.get('model_id')
        if model_id:
            queryset = queryset.filter(model_id=model_id)
        
        return queryset
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Extract validated data
        validated_data = serializer.validated_data
        content = validated_data.pop('content')
        
        # Create embedding
        embedding = create_embedding_for_text(
            text=content,
            project_id=validated_data['project'].id,
            name=validated_data.get('name'),
            description=validated_data.get('description', ''),
            entity_type=validated_data.get('entity_type', 'text'),
            metadata=validated_data.get('metadata', {}),
            dataset=validated_data.get('dataset'),
            model=validated_data.get('model')
        )
        
        # Return the created embedding
        result_serializer = VectorEmbeddingSerializer(embedding)
        return Response(result_serializer.data, status=status.HTTP_201_CREATED)

class VectorIndexViewSet(viewsets.ModelViewSet):
    queryset = VectorIndex.objects.all()
    
    def get_serializer_class(self):
        if self.action == 'create':
            return VectorIndexCreateSerializer
        return VectorIndexSerializer
    
    def get_queryset(self):
        queryset = VectorIndex.objects.all()
        
        # Filter by project
        project_id = self.request.query_params.get('project_id')
        if project_id:
            queryset = queryset.filter(project_id=project_id)
        
        # Filter by index type
        index_type = self.request.query_params.get('index_type')
        if index_type:
            queryset = queryset.filter(index_type=index_type)
        
        return queryset
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Extract validated data
        validated_data = serializer.validated_data
        embedding_ids = validated_data.pop('embedding_ids', [])
        
        # Get the embeddings
        embeddings = VectorEmbedding.objects.filter(id__in=embedding_ids)
        
        if not embeddings:
            return Response(
                {"error": "No valid embeddings found with the provided IDs"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create the index
        index = create_index_for_embeddings(
            embeddings=embeddings,
            index_type=validated_data.get('index_type', 'flat'),
            name=validated_data.get('name'),
            description=validated_data.get('description', '')
        )
        
        # Return the created index
        result_serializer = VectorIndexSerializer(index)
        return Response(result_serializer.data, status=status.HTTP_201_CREATED)

class VectorSearchQueryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = VectorSearchQuery.objects.all().order_by('-created_at')
    serializer_class = VectorSearchQuerySerializer
    
    def get_queryset(self):
        queryset = VectorSearchQuery.objects.all().order_by('-created_at')
        
        # Filter by project
        project_id = self.request.query_params.get('project_id')
        if project_id:
            queryset = queryset.filter(project_id=project_id)
        
        return queryset

class VectorSimilarityViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = VectorSimilarity.objects.all()
    serializer_class = VectorSimilaritySerializer
    
    def get_queryset(self):
        queryset = VectorSimilarity.objects.all()
        
        # Filter by source embedding
        source_id = self.request.query_params.get('source_id')
        if source_id:
            queryset = queryset.filter(source_embedding_id=source_id)
        
        # Filter by target embedding
        target_id = self.request.query_params.get('target_id')
        if target_id:
            queryset = queryset.filter(target_embedding_id=target_id)
        
        # Filter by minimum similarity score
        min_score = self.request.query_params.get('min_score')
        if min_score:
            try:
                min_score = float(min_score)
                queryset = queryset.filter(similarity_score__gte=min_score)
            except ValueError:
                pass
        
        return queryset

class VectorSearchAPIView(generics.GenericAPIView):
    serializer_class = VectorSearchRequestSerializer
    
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Extract validated data
        query = serializer.validated_data['query']
        index_id = serializer.validated_data['index_id']
        top_k = serializer.validated_data.get('top_k', 10)
        filters = serializer.validated_data.get('filters', {})
        
        try:
            # Perform the search
            search_results = search_vectors(
                query_text=query,
                index_id=index_id,
                top_k=top_k
            )
            
            # Serialize the results
            result_serializer = VectorSearchResultSerializer(
                search_results['results'], 
                many=True
            )
            
            # Return the search results
            return Response({
                'results': result_serializer.data,
                'query_time_ms': search_results['query_time_ms'],
                'total_results': search_results['total_results']
            })
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
