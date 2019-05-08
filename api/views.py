import json

from django.contrib.auth.models import User, AnonymousUser

from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework.decorators import permission_classes, action
from rest_framework.authtoken.models import Token

from api.models import (
    Doctor,
    Patient,
    CellCategory,
    City,
    CellExtraction,
    Cell,
    Slide,
    Anamnesis
)

from api.serializers import (
    DoctorSerializer,
    PatientSerializer,
    CellCategorySerializer,
    CitySerializer,
    CellExtractionSerializer,
    CellSerializer,
    SlideSerializer,
    AnamnesisSerializer
)
from api.permissions import IsPatientOwner
from django.http.request import QueryDict
from django.shortcuts import _get_queryset


class UserViewSet(ModelViewSet):
    queryset = User.objects.all()
    serializer_class = User.objects.all()


class DoctorViewSet(ModelViewSet):
    queryset = Doctor.objects.all()
    serializer_class = DoctorSerializer

    def list(self, request, *args, **kwargs):
        print(request.GET)
        token = request.GET.get('token')
        if token != None:
            user = Token.objects.get(key=token).user
            queryset = Doctor.objects.get(user=user)
            serializer = self.get_serializer(queryset, many=False)
        else:
            queryset = self.get_queryset()
            serializer = self.get_serializer(queryset, many=True)

        return Response(serializer.data)

    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        instance.user.username = request.data.get('username')
        instance.user.email = request.data.get('email')
        instance.user.first_name = request.data.get('first_name')
        instance.user.last_name = request.data.get('last_name')
        instance.save()

        serializer = self.get_serializer(instance)
        serializer.is_valid(raise_exception=True)

        self.perform_update(serializer)

        return Response(serializer)


@permission_classes((IsPatientOwner, ))
class PatientViewSet(ModelViewSet):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer

    def get_queryset(self):
        print('get_queryset()')
        user = self.request.user

        request = Doctor.objects.none();
        if user.is_staff:
            request = Patient.objects.all()
        else:
            doctor = Doctor.objects.get(user=user)
            request = doctor.getPatients()

        return request


class CellCategoryViewSet(ModelViewSet):
    queryset = CellCategory.objects.all()
    serializer_class = CellCategorySerializer


class CityViewSet(ModelViewSet):
    queryset = City.objects.all()
    serializer_class = CitySerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['code', 'name', 'province_code']


class CellExtractionViewSet(ModelViewSet):
    queryset = CellExtraction.objects.all()
    serializer_class = CellExtractionSerializer

    def get_queryset(self):
        user = self.request.user

        queryset = CellExtraction.objects.none()
        if user.is_staff:
            queryset = CellExtraction.objects.all()
        elif not isinstance(user, AnonymousUser):
            print('User:', user);
            import logging
            logger = logging.getLogger(__name__)
            logger.error('User:', user)
            doctor = Doctor.objects.get(user=user)
            queryset = CellExtraction.objects.all().filter(doctor=doctor)

        queryset = queryset.order_by('-extraction_date')
        return queryset

    def list(self, request, *args, **kwargs):
        patient = request.query_params.get('patient')

        if patient != None:
            patient = Patient.objects.get(id=patient)
            queryset = self.get_queryset().filter(patient=patient)
            serializer = self.get_serializer(queryset, many=True)
            return Response(serializer.data)

        return ModelViewSet.list(self, request, *args, **kwargs)

    def create(self, request, *args, **kwargs):
        if isinstance(request.data, QueryDict):
            _mutable = request.data._mutable
            request.data._mutable = True

        request.data['doctor'] = Doctor.objects.get(user=request.user).id

        if isinstance(request.data, QueryDict):
            request.data._mutable = _mutable

        return ModelViewSet.create(self, request, *args, **kwargs)

    @action(detail=False, methods=['get'])
    def last(self, request):
        doctor = Doctor.objects.get(user=request.user)
        patient = Patient.objects.get(id=request.GET['patient'])

        print('Doctor =', doctor)
        print('Patient =', patient)

        last_extraction = CellExtraction.objects.filter(doctor=doctor).last()

        serializer = self.get_serializer(last_extraction)

        return Response(serializer.data)


class CellViewSet(ModelViewSet):
    queryset = Cell.objects.all()
    serializer_class = CellSerializer

class SlideViewSet(ModelViewSet):
    queryset = Slide.objects.all()
    serializer_class = SlideSerializer

class AnamnesisViewSet(ModelViewSet):
    queryset = Anamnesis.objects.all()
    serializer_class = AnamnesisSerializer
