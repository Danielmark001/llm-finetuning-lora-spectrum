from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ProjectViewSet, DatasetViewSet, ModelViewSet, EvaluationViewSet

router = DefaultRouter()
router.register(r'projects', ProjectViewSet)
router.register(r'datasets', DatasetViewSet)
router.register(r'models', ModelViewSet)
router.register(r'evaluations', EvaluationViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
