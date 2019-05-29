from django.urls import path

from media.views import cell_image_view, slide_image_view, report_view


urlpatterns = [
    path('cells/<str:filename>', cell_image_view, name='cell-image`'),
    path('slides/<str:filename>', slide_image_view, name='slide-image'),
    path('report/<str:filename>', report_view, name='report-pdf')
]
