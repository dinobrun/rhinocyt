from django.contrib import admin

from api.models import (
    Doctor,
    Patient,
    CellCategory,
    City,
    CellExtraction,
    Cell
)


admin.site.register(CellCategory)
admin.site.register(CellExtraction)
admin.site.register(Cell)

admin.site.register(Doctor)
admin.site.register(Patient)

admin.site.register(City)