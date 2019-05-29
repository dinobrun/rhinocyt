import os

from django.http.response import HttpResponse

from api.models import Cell, Patient, Slide, Report


def cell_image_view(request, filename):
#     if not request.user:
#         raise Exception("The user must be authenticated")

    path = os.path.join(Patient.PATH_CELLS, filename)

    cell = Cell.objects.get(image=path)
    image_data = cell.image.open().read()

    return HttpResponse(image_data, content_type='image/png')

def slide_image_view(request, filename):
    path = os.path.join('slides', filename)
    slide = Slide.objects.get(image=path)
    image_data = slide.image.open().read()

    return HttpResponse(image_data, content_type='image/png')

def report_view(request, filename):
    path = os.path.join('report', filename)
    report = Report.objects.get(report_file=path)
    report_data = report.report_file.open().read()

    return HttpResponse(report_data, content_type='application/pdf')
