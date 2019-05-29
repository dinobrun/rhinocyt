from django.urls.conf import path, include

from rest_framework.routers import DefaultRouter
from rest_framework.authtoken.views import obtain_auth_token

from api.views import (
    DoctorViewSet,
    PatientViewSet,
    CellCategoryViewSet,
    CityViewSet,
    CellExtractionViewSet,
    CellViewSet,
    SlideViewSet,
    AnamnesisViewSet,
    DiagnosisViewSet,
    DiagnosisExtractionViewSet,
    PrickTestViewSet,
    ReportViewSet,
    AllergyViewSet
)


# API router
router = DefaultRouter()
# TODO: FIX: 'QuerySet' object is not callable
# router.register('users', UserViewSet)
router.register('doctors', DoctorViewSet)
router.register('patients', PatientViewSet)
router.register('cells', CellViewSet)
router.register('cell-categories', CellCategoryViewSet)
router.register('cell-extractions', CellExtractionViewSet)
router.register('slides', SlideViewSet)
router.register('cities', CityViewSet)
router.register('anamnesis', AnamnesisViewSet)
router.register('diagnosis-extractions', DiagnosisExtractionViewSet)
router.register('diagnosis', DiagnosisViewSet)
router.register('prick-test', PrickTestViewSet)
router.register('allergy', AllergyViewSet)
router.register('report', ReportViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('token-auth/', obtain_auth_token, name='api_token_auth')
]
