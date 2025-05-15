from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Project, Dataset, Model, Evaluation
from .serializers import ProjectSerializer, DatasetSerializer, ModelSerializer, EvaluationSerializer

class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    
    def get_queryset(self):
        queryset = Dataset.objects.all()
        project_id = self.request.query_params.get('project_id', None)
        if project_id is not None:
            queryset = queryset.filter(project__id=project_id)
        return queryset

class ModelViewSet(viewsets.ModelViewSet):
    queryset = Model.objects.all()
    serializer_class = ModelSerializer
    
    def get_queryset(self):
        queryset = Model.objects.all()
        project_id = self.request.query_params.get('project_id', None)
        if project_id is not None:
            queryset = queryset.filter(project__id=project_id)
        return queryset

class EvaluationViewSet(viewsets.ModelViewSet):
    queryset = Evaluation.objects.all()
    serializer_class = EvaluationSerializer
    
    def get_queryset(self):
        queryset = Evaluation.objects.all()
        model_id = self.request.query_params.get('model_id', None)
        if model_id is not None:
            queryset = queryset.filter(model__id=model_id)
        return queryset
