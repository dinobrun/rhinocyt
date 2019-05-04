from django.urls.conf import path

from app.views import home


urlpatterns = [
    path('', home, name='home')
]