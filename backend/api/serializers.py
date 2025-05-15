from rest_framework import serializers
from .models import Project, Dataset, Model, Evaluation

class EvaluationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Evaluation
        fields = ['id', 'name', 'description', 'metrics', 'created_at', 'model']

class ModelSerializer(serializers.ModelSerializer):
    evaluations = EvaluationSerializer(many=True, read_only=True)
    
    class Meta:
        model = Model
        fields = ['id', 'name', 'description', 'type', 'created_at', 'project', 'evaluations']

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['id', 'name', 'description', 'file_path', 'created_at', 'project']

class ProjectSerializer(serializers.ModelSerializer):
    datasets = DatasetSerializer(many=True, read_only=True)
    models = ModelSerializer(many=True, read_only=True)
    
    class Meta:
        model = Project
        fields = ['id', 'name', 'description', 'created_at', 'updated_at', 'datasets', 'models']
