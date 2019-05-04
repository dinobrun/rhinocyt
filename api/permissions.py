from rest_framework.permissions import (
    BasePermission,
    SAFE_METHODS,
    IsAuthenticated
)

from api.models import Doctor


class IsDoctorOwner(IsAuthenticated):
    def has_object_permission(self, request, view, obj):
        return request.user == obj.user or request.user.is_staff
   

class IsPatientOwner(IsAuthenticated):
    def has_object_permission(self, request, view, obj):
        has_permission = False
        
        if request.user.is_staff:
            has_permission = True
        else:
            try:
                doctor = Doctor.objects.get(user=request.user)
                has_permission = obj.doctor == doctor
            except Doctor.DoesNotExist:
                print("Doctor doesn't exists.")

        print('Has permission:', has_permission)
        return has_permission

    
class IsAdminOrReadOnly(BasePermission):
    def has_permission(self, request, view):
        return request.method in SAFE_METHODS or (request.user and request.user.is_staff)
