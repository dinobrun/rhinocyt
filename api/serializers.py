from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password

from rest_framework.serializers import ModelSerializer
from rest_framework import serializers
from rest_framework.authtoken.models import Token
# from rest_framework.fields import SerializerMethodField

from api.models import (
    Doctor,
    Patient,
    CellCategory,
    City,
    CellExtraction,
    Cell,
    Slide,
    Anamnesis,
    DiagnosisExtraction,
    Diagnosis,
    Allergy,
    Report,
    PrickTest
)


class CellCategorySerializer(ModelSerializer):
    class Meta:
        model = CellCategory
        fields = '__all__'


class CitySerializer(ModelSerializer):
    class Meta:
        model = City
        fields = '__all__'


class UserSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = ('username', 'email', 'password', 'first_name', 'last_name')
        extra_kwargs = {'password': {'write_only': True}}

    def validate_password(self, value):
        return make_password(value)


class PatientSerializer(ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'


class DoctorSerializer(ModelSerializer):
    last_name = serializers.CharField(source='user.last_name')
    first_name = serializers.CharField(source='user.first_name')
    username = serializers.CharField(source='user.username')
    email = serializers.CharField(source='user.email')
    password = serializers.CharField(source='user.password', write_only=True)
    patients = PatientSerializer(many=True, read_only=True)

    class Meta:
        model = Doctor
        fields = ('id', 'last_name', 'first_name', 'username', 'email', 'password', 'patients')
        depth = 1

    def create(self, validated_data):
        user_data = validated_data['user']
        user = User.objects.create_user(last_name=user_data['last_name'],
                                   first_name=user_data['first_name'],
                                   username=user_data['username'],
                                   email=user_data['email'],
                                   password=user_data['password'])
        Token.objects.create(user=user)
        return Doctor.objects.create(user=user)


class CellSerializer(ModelSerializer):
    class Meta:
        model = Cell
        fields = '__all__'


class SlideSerializer(ModelSerializer):
    cells = CellSerializer(many=True, read_only=True)
    cell_extraction_id = serializers.SerializerMethodField()
    patient_id = serializers.SerializerMethodField()

    class Meta:
        model = Slide
        fields = ['id', 'image', 'cell_extraction', 'cell_extraction_id', 'patient', 'patient_id', 'cells']
        extra_kwargs = {
            'cell_extraction': {'write_only': True},
            'patient': {'write_only': True}
        }

    def get_cell_extraction_id(self, obj):
        return obj.cell_extraction.id

    def get_patient_id(self, obj):
        return obj.patient.id


class CellExtractionSerializer(ModelSerializer):
    slides = SlideSerializer(many=True, read_only=True)

    class Meta:
        model = CellExtraction
        fields = '__all__'

class PrickTestSerializer(ModelSerializer):
    class Meta:
        model = PrickTest
        fields = '__all__'

class AnamnesisSerializer(ModelSerializer):
    prick_test = PrickTestSerializer(many=True, read_only=True)
    class Meta:
        model = Anamnesis
        fields = '__all__'

class DiagnosisSerializer(ModelSerializer):
    class Meta:
        model = Diagnosis
        fields = '__all__'

class DiagnosisExtractionSerializer(ModelSerializer):
    diagnosis = DiagnosisSerializer(many=True, read_only=True)

    class Meta:
        model = DiagnosisExtraction
        fields = '__all__'

class AllergySerializer(ModelSerializer):
    class Meta:
        model = Allergy
        fields = '__all__'

class ReportSerializer(ModelSerializer):
    class Meta:
        model = Report
        fields = '__all__'
