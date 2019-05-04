import os
import string
import random

from pathlib import Path

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinLengthValidator

from rhinocyt import settings


class ApiModel(models.Model):
    """
    Abstract API model for the Cynochew models.
    """
    
    class Meta:
        abstract = True
    
    def __str__(self):
        return '{0}[{1}]'.format(self.__class__.__name__, self.id)


class CellCategory(ApiModel):
    classnum = models.SmallIntegerField()
    name = models.CharField(max_length=30)
    
    class Meta:
        verbose_name_plural = 'cell categories'


class City(models.Model):
    code = models.CharField(primary_key=True, max_length=6, validators=[MinLengthValidator(6, 'Length has to be 6.')])
    name = models.CharField(max_length=100)
    province_code = models.CharField(max_length=2, validators=[MinLengthValidator(2, 'Length has to be 2.')])
    
    class Meta:
        verbose_name_plural = 'cities'


class Doctor(ApiModel):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # other fields
    
    def getPatients(self):
        return self.patients.all()
    
    def __str__(self):
        return '{0}: {1} {2}'.format(ApiModel.__str__(self), self.user.last_name, self.user.first_name)


class Patient(ApiModel):
    # Patient's paths
    PATH_BASE = str(Path.home()) # probably somewhere else.
    PATH_DATA = 'data'
    PATH_INPUTS = 'inputs'
    PATH_CELLS = 'cells'
    
    # Fields
    first_name = models.CharField(max_length=25, blank=True, default='')
    last_name = models.CharField(max_length=25, blank=True, default='')
    fiscal_code = models.CharField(max_length=16, blank=True, default='', validators=[MinLengthValidator(16, message='Length has to be 16.')])
    residence_city = models.ForeignKey(City, on_delete=models.SET_NULL, related_name='residence_city', blank=True, null=True)
    birth_city = models.ForeignKey(City, on_delete=models.SET_NULL, related_name='birth_city', blank=True, null=True)
    birthdate = models.DateField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)
    doctor = models.ForeignKey(Doctor, related_name='patients', on_delete=models.CASCADE)

    
    def getDataPath(self):
        """
        Returns the data path of the patient.
        """
        if id == None:
            raise Exception('The patient must have a valid id.')
        
        return os.path.join(self.PATH_BASE, self.PATH_DATA, str(self.id))
    
    def getInputsPath(self):
        """
        Returns the inputs path of the patient, where all the input
        are located.
        """
        return os.path.join(self.getDataPath(), self.PATH_INPUTS)
    
    def getCellsPath(self):
        """
        Returns the cell input path of the patient, where the extracted
        cells are located.
        """
        return os.path.join(self.getInputsPath(), self.PATH_CELLS)

    def getCellCategoryPath(self, classnum):
        """
        TODO, if can be useful.
        """
        pass
    
    def __str__(self):
        return '{0}: {1} {2}'.format(ApiModel.__str__(self), self.last_name, self.first_name)


class CellExtraction(ApiModel):
    doctor = models.ForeignKey(Doctor, related_name='cell_extractions', on_delete=models.CASCADE)
    patient = models.ForeignKey(Patient, related_name='cell_extractions', on_delete=models.CASCADE)
    extraction_date = models.DateTimeField(auto_now_add=True)

class Slide(ApiModel):
    image = models.ImageField(upload_to='slides/')
    cell_extraction = models.ForeignKey(CellExtraction, related_name='slides', on_delete=models.CASCADE)
    patient = models.ForeignKey(Patient, related_name='slides', on_delete=models.CASCADE)
    
    def save(self, *args, **kwargs):
        """
        Save and extract the cells from the slide.
        """
        super().save(*args, **kwargs);
        self.extract_cells()
    
    def get_cell_regions(self):
        """
        Detects and returns the cell regions from a
        slide.
        """
        import cv2
        import numpy as np
        from skimage import morphology
        from skimage.feature import peak_local_max
        from scipy import ndimage
        
        # defining some constants
        SPATIAL_WINDOW_RADIUS = 21
        COLOR_WINDOW_RADIUS = 51
        DISK_RADIUS = 5
        MIN_PEAK_DISTANCE = 20
        MIN_REGION_SIZE = 1000
        MAX_REGION_SIZE = 15000

        # using cv2 to read the image
        image = cv2.imread(self.image.file.name)
        
        # perform pyramid mean shift filtering
        # to aid the thresholding step
        shifted = cv2.pyrMeanShiftFiltering(image, SPATIAL_WINDOW_RADIUS, COLOR_WINDOW_RADIUS)
         
        # convert the mean shift image to grayscale, then apply
        # Otsu's thresholding
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
    
    def extract_cells(self):
        """
        Extracts and saves the cells from a slide.
        """
        import cv2
        from skimage.measure._regionprops import regionprops
        from skimage import io
        from matplotlib import patches
        
        image_path = self.image.file.name
        image_extension = image_path.split('.')[-1]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        filtered_labels = self.get_cell_regions()
        region_list = regionprops(filtered_labels)[1:]
        for region in region_list:
            # draw circle around cells
#             patches.Circle(region.centroid, radius=region.equivalent_diameter, fill=False, edgecolor='red', linewidth=2)
        
            # Transform the region to crop from rectangular to square
            minr, minc, maxr, maxc = region.bbox
            x_side = maxc - minc
            y_side = maxr - minr
            
            if x_side > y_side:
                maxr = x_side + minr
            else:
                maxc = y_side + minc
    
            if (minc > Cell.IMAGE_PADDING) & (minr > Cell.IMAGE_PADDING):
                minc -= Cell.IMAGE_PADDING
                minr -= Cell.IMAGE_PADDING
            
            maxr += Cell.IMAGE_PADDING
            maxc += Cell.IMAGE_PADDING
            
            cell = image[minr:maxr, minc:maxc]
            
            cell_image_filename = Cell.generateFilename(image_extension)
            cells_path = Cell.getMediaPath()
            
            # create the cells path if doesn't exists
            if not os.path.isdir(cells_path):
                os.mkdir(cells_path)
            
            cell_image_path = os.path.join(cells_path, cell_image_filename)
            io.imsave(cell_image_path, cell)

            cell_category = Cell.predict_class(cell_image_path)
            Cell.objects.create(patient=self.patient, slide=self, cell_category=cell_category, image=Cell.getCellRelativePath(cell_image_filename))
            


class Cell(ApiModel):
    # model vars
    isModelLoaded = False
    MODEL_FILENAME = os.path.join(settings.PROJECT_PATH, 'model.h5')
    MODEL_WEIGHT_FILENAME = os.path.join(settings.PROJECT_PATH, 'model_weights.hdf5')
    
    # the image padding of the cell
    IMAGE_PADDING = 20
    
    # the media folder name
    MEDIA_FOLDER = 'cells'
    
    # fields
    patient = models.ForeignKey(Patient, related_name='cells', on_delete=models.CASCADE)
    slide = models.ForeignKey(Slide, related_name='cells', on_delete=models.CASCADE)
    cell_category = models.ForeignKey(CellCategory, related_name='cells', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='cells/')
    validated = models.BooleanField(blank=True, default=False)

    @staticmethod
    def predict_class(cell_image_path):
        import cv2
        import numpy as np

        Cell.loadModel()

        image = cv2.imread(cell_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, np.float32)
        image = cv2.resize(image, (50, 50))
        image /= 255
        image = image.reshape((1,) + image.shape)

        cell_class = int(Cell.MODEL.predict_classes(image))
        return CellCategory.objects.get(classnum=cell_class)
    
    @staticmethod
    def loadModel():
        if Cell.isModelLoaded == False:
            from keras.models import load_model
            
            # Model construction
            Cell.MODEL = load_model(Cell.MODEL_FILENAME)
            Cell.MODEL.load_weights(Cell.MODEL_WEIGHT_FILENAME)
            Cell.MODEL.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        
            Cell.isModelLoaded = True
    
    @staticmethod
    def generateFilename(image_extension):
        return '{0}.{1}'.format(''.join(random.choice(string.ascii_letters + string.digits) for x in range(32)), image_extension)  # @UnusedVariable

    @staticmethod
    def getMediaPath():
        return os.path.join(settings.MEDIA_ROOT, Cell.MEDIA_FOLDER)

    @staticmethod
    def getCellRelativePath(filename):
        return os.path.join(Cell.MEDIA_FOLDER, filename)

