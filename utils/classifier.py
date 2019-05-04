#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import shutil
import glob

from keras.models import load_model

from api.models import Patient, CellCategory

# Model files
MODEL_FILENAME = 'model.h5'
MODEL_WEIGHT_FILENAME = 'model_weights.hdf5'

# Model construction
MODEL = load_model(MODEL_FILENAME)
MODEL.load_weights(MODEL_WEIGHT_FILENAME)
MODEL.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Default class is "other"
CLASS_CELL_DEFAULT = -1
CLASS_CELL_LIST = CellCategory.objects.values_list('classnum', flat=True)

# Cell image directory name
CELL_IMAGES_DIRNAME = "cells"

def load_data(data_directory):
	"""
	Returns the image data from a given data directory.
	:param data_directory The directory of the data.
	:return               The image list and the file path list.
	"""
	
	images_path = os.path.join(data_directory, Patient.PATH_CELLS)
	file_path_list = []
	image_list = []

	if os.path.isdir(images_path):
		for image_file in glob.iglob(os.path.join(images_path, '*.png')):
			file_path = os.path.join(images_path, image_file)
			file_path_list.append(file_path)

			image = cv2.imread(file_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = cv2.resize(image, (50, 50))
			image_list.append(image)

	return image_list, file_path_list

def predict_patient_cells(patient: Patient):
	"""
	Predicts the (new) cells of a given patient.
	:param patient The patient.
	"""
	
	input_data_path = patient.getInputsPath()

	image_list, file_path_list = load_data(input_data_path)
	image_list = np.array(image_list, np.float32);
	image_list /= 255
	image_list = image_list.reshape(len(image_list), 50, 50, 3)

	for i in range(len(image_list)):
		image = image_list[i]
		file_path = file_path_list[i]

		image = image.reshape((1,) + image.shape)

		cell_class = int(MODEL.predict_classes(image))
		cell_category = CellCategory.objects.get(classnum=cell_class)
		cell_path =  os.path.join(input_data_path, cell_category.name)
		
		if not os.path.isdir(cell_path):
			os.mkdir(cell_path)
		
		shutil.move(file_path, cell_path)
		print('Image "{}" is a "{}"'.format(file_path, cell_category.name))