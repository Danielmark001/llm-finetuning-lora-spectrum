from django.contrib import admin
from .models import Project, Dataset, Model, Evaluation

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at', 'updated_at')
    search_fields = ('name', 'description')

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'project', 'created_at')
    list_filter = ('project',)
    search_fields = ('name', 'description')

@admin.register(Model)
class ModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'type', 'project', 'created_at')
    list_filter = ('project', 'type')
    search_fields = ('name', 'description')

@admin.register(Evaluation)
class EvaluationAdmin(admin.ModelAdmin):
    list_display = ('name', 'model', 'created_at')
    list_filter = ('model__project', 'model')
    search_fields = ('name', 'description')
