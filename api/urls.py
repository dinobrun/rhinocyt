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
    SlideViewSet
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

urlpatterns = [
    path('', include(router.urls)),
    path('auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('token-auth/', obtain_auth_token, name='api_token_auth')
]

