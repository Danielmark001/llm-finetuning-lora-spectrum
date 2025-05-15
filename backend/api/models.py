from django.db import models

class Project(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name

class Dataset(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    file_path = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    project = models.ForeignKey(Project, related_name='datasets', on_delete=models.CASCADE)
    
    def __str__(self):
        return self.name

class Model(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    type = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    project = models.ForeignKey(Project, related_name='models', on_delete=models.CASCADE)
    
    def __str__(self):
        return self.name

class Evaluation(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    metrics = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    model = models.ForeignKey(Model, related_name='evaluations', on_delete=models.CASCADE)
    
    def __str__(self):
        return self.name
