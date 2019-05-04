#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from api.models import City, CellCategory, Doctor


DATABASE_FILE = 'db.sqlite3'

CITIES_FILE = 'cities.json'

CELL_CATEGORIES = [
    CellCategory(id=1, name='epithelium', classnum=0),
    CellCategory(id=2, name='neutrophil', classnum=1),
    CellCategory(id=3, name='eosinophil', classnum=2),
    CellCategory(id=4, name='mastocyte', classnum=3),
    CellCategory(id=5, name='lymphocyte', classnum=4),
    CellCategory(id=6, name='mucipara', classnum=5),
    CellCategory(id=7, name='other', classnum=6),
]

def default():
    """
    Updates all the cities based on the file of cities.
    """

    # Default cell categories
    for cell_category in CELL_CATEGORIES:
        cell_category.save()
    
    # Default cities
    response = input('Sync cities? [y/N]: ')
    if response == 'y':
        with open(CITIES_FILE) as json_data:
            data = json.load(json_data)
            
            for city_data in data:
                code = city_data['codice']
                name = city_data['nome']
                province_code = city_data['sigla']
                
                city = City(code=code, name=name, province_code=province_code)
                city.save()
