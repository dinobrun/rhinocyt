#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------
# Cells extraction using the watershed method from
# OpenCv library
# ---------------------------------------------------

import glob
import os
import cv2

import numpy as np
import matplotlib.patches as mpatches

from scipy import ndimage

from skimage import morphology
from skimage import io
from skimage.measure import regionprops
from skimage.feature import peak_local_max

from api.models import Patient

# Data for mean shift filtering
SPATIAL_WINDOW_RADIUS = 21
COLOR_WINDOW_RADIUS = 51

# Morphology's disk radius
DISK_RADIUS = 5

# The minimum peak distance
MIN_PEAK_DISTANCE = 20

# The min/max region size
MIN_REGION_SIZE = 1000
MAX_REGION_SIZE = 15000

#padding of each cell extracted
CELL_EXTRACTION_PADDING = 20

#cell image extension
CELL_IMAGE_EXTENSION = 'png'

def cell_detection(image):

    # perform pyramid mean shift filtering
    # to aid the thresholding step
    shifted = cv2.pyrMeanShiftFiltering(image, SPATIAL_WINDOW_RADIUS, COLOR_WINDOW_RADIUS)

    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    # gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # morphological transformation
    selem = morphology.disk(DISK_RADIUS)
    thresh = morphology.dilation(thresh, selem)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    d = ndimage.distance_transform_edt(thresh)
    local_max = peak_local_max(d, MIN_PEAK_DISTANCE, indices=False, labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    labels = morphology.watershed(-d, markers, mask=thresh)

    # remove labels too small.
    filtered_labels = np.copy(labels)
    component_sizes = np.bincount(labels.ravel())
    too_small = component_sizes < MIN_REGION_SIZE
    too_small_mask = too_small[labels]
    filtered_labels[too_small_mask] = 1

    # remove labels that are too big.
    too_big = component_sizes > MAX_REGION_SIZE
    too_big_mask = too_big[labels]
    filtered_labels[too_big_mask] = 1

    return filtered_labels


def cell_extraction(image, imagenum, patient):
    
    filtered_labels = cell_detection(image)

    i = 0
    region_list = regionprops(filtered_labels)[1:]

    for region in region_list:
        # draw circle around cells
        mpatches.Circle(region.centroid, radius=region.equivalent_diameter, fill=False, edgecolor='red', linewidth=2)
        
        # Transform the region to crop from rectangular to square
        minr, minc, maxr, maxc = region.bbox
        x_side = maxc - minc
        y_side = maxr - minr
        if x_side > y_side:
            maxr = x_side + minr
        else:
            maxc = y_side + minc

        if (minc > CELL_EXTRACTION_PADDING) & (minr > CELL_EXTRACTION_PADDING):
            minc -= CELL_EXTRACTION_PADDING
            minr -= CELL_EXTRACTION_PADDING
        maxr += CELL_EXTRACTION_PADDING
        maxc += CELL_EXTRACTION_PADDING
        
        cell = image[minr:maxr, minc:maxc]

        image_filename = 'img{}_cell{}.{}'.format(imagenum, i, CELL_IMAGE_EXTENSION)
        
        cells_path = patient.getCellsPath()
        if not os.path.isdir(cells_path):
            os.mkdir(cells_path)
        
        cell_image_path = os.path.join(cells_path, image_filename)
        io.imsave(cell_image_path, cell)

        print('Image saved on "{}"'.format(cell_image_path))

        i += 1

def extract(patient: Patient, imagedir):
    imagenum = 0
    file_image_iterator = glob.iglob(os.path.join(imagedir), '*.{}'.format(CELL_IMAGE_EXTENSION))
    
    for file in file_image_iterator:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            cell_extraction(image, imagenum, patient)
        except ValueError:
            continue #handle this exception 
        
        imagenum += 1
