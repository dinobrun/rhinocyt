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
    epithelium_num = models.SmallIntegerField(default=0)
    neutrophil_num = models.SmallIntegerField(default=0)
    eosinophil_num = models.SmallIntegerField(default=0)
    mastocyte_num = models.SmallIntegerField(default=0)
    lymphocyte_num = models.SmallIntegerField(default=0)
    mucipara_num = models.SmallIntegerField(default=0)

    def get_cell_grade(cell_category):
        """
        Return the grade of a cell category in the cell extraction
        """
        if cell_category == "mucipara":
            return count300(mucipara_num)
        elif cell_category == "epithelium":
            return count300(epithelium_num)
        elif cell_category == "neutrophil":
            return count100(neutrophil_num)
        elif cell_category == "eosinophil":
            return count30(eosinophil_num)
        elif cell_category == "mastocyte":
            return count30(mastocyte_num)
        elif cell_category == "lymphocyte":
            return count30(lymphocyte_num)
        else:
            return 0

    def count300(cell_number):
        if cell_number == 0:
            return 0
        elif cell_number > 0 and cell_number < 101:
            return 1
        elif cell_number > 100 and cell_number < 201:
            return 2
        elif cell_number > 200 and cell_number < 301:
            return 3
        elif cell_number > 300:
            return 4
        else:
            return 0

    def count100(cell_number):
        if cell_number == 0:
            return 0
        elif cell_number > 0 and cell_number < 21:
            return 1
        elif cell_number > 20 and cell_number < 41:
            return 2
        elif cell_number > 40 and cell_number < 101:
            return 3
        elif cell_number > 100:
            return 4
        else:
            return 0

    def count30(cell_number):
        if cell_number == 0:
            return 0
        elif cell_number > 0 and cell_number < 6:
            return 1
        elif cell_number > 5 and cell_number < 11:
            return 2
        elif cell_number > 10 and cell_number < 31:
            return 3
        elif cell_number > 30:
            return 4
        else:
            return 0

    def count16(cell_number):
        if cell_number == 0:
            return 0
        elif cell_number > 0 and cell_number < 4:
            return 1
        elif cell_number > 3 and cell_number < 8:
            return 2
        elif cell_number > 7 and cell_number < 17:
            return 3
        elif cell_number > 16:
            return 4
        else:
            return 0



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

    def save(self, *args, **kwargs):
        """
        Get reference for the cell extraction and update the number of cell per type
        """
        slide_ref = Slide.objects.get(pk=self.slide.pk)
        cell_extract_ref = CellExtraction.objects.get(pk=slide_ref.cell_extraction.pk)
        #update the number of cells per type in cell_extraction reference
        if self.cell_category == CellCategory.objects.get(name="epithelium"):
            cell_extract_ref.epithelium_num += 1
        elif self.cell_category == CellCategory.objects.get(name="neutrophil"):
            cell_extract_ref.neutrophil_num += 1
        elif self.cell_category == CellCategory.objects.get(name="eosinophil"):
            cell_extract_ref.eosinophil_num += 1
        elif self.cell_category == CellCategory.objects.get(name="mastocyte"):
            cell_extract_ref.mastocyte_num += 1
        elif self.cell_category == CellCategory.objects.get(name="lymphocyte"):
            cell_extract_ref.lymphocye_num += 1
        elif self.cell_category == CellCategory.objects.get(name="mucipara"):
            cell_extract_ref.mucipara_num += 1
        cell_extract_ref.save()
        super().save(*args, **kwargs);

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

class Anamnesis(ApiModel):
    ALLERG_TYPE = (
        ('AL', 'alimenti'),
        ('IN', 'inalanti')
    )
    #Anamnesi familiare
    patient = models.ForeignKey(Patient, related_name='anamnsesis', on_delete=models.CASCADE)
    anamnesis_date = models.DateTimeField(auto_now_add=True)
    allergy_gen = models.CharField(max_length=2, choices=ALLERG_TYPE, null=True)
    allergy_fra = models.CharField(max_length=2, choices=ALLERG_TYPE, null=True)
    polip_nas_gen = models.BooleanField(blank=True, default=False)
    polip_nas_fra = models.BooleanField(blank=True, default=False)
    asma_bronch_gen = models.BooleanField(blank=True, default=False)
    asma_bronch_fra = models.BooleanField(blank=True, default=False)

    #Sintomatologia
    LEFT_RIGHT_TYPE = (
        ('SX', 'sinistra'),
        ('DX', 'destra'),
        ('EX', 'bilaterale')
    )
    RINORREA_ESSUDATO_TYPE = (
        ('SI', 'sieroso'),
        ('MU', 'mucoso'),
        ('PU', 'purulento'),
        ('EM', 'ematico')
    )
    STARNUTAZIONE_TYPE = (
        ('SP', 'sporadica'),
        ('AS', 'a salve')
    )
    PROB_OLF_TYPE = (
        ('IP', 'iposmia'),
        ('AN', 'anosmia'),
        ('CA', 'cacosimia')
    )
    SINDR_VER_TYPE = (
        ('SO', 'soggettiva'),
        ('OG', 'oggettiva')
    )

    ostr_nas = models.CharField(max_length=2, choices=LEFT_RIGHT_TYPE, null=True)
    rinorrea = models.CharField(max_length=2, choices=RINORREA_ESSUDATO_TYPE, null=True)
    prur_nas = models.BooleanField(blank=True, default=False)
    starnutazione = models.CharField(max_length=2, choices=STARNUTAZIONE_TYPE, null=True)
    prob_olf = models.CharField(max_length=2, choices=PROB_OLF_TYPE, null=True)
    ovatt_aur = models.CharField(max_length=2, choices=LEFT_RIGHT_TYPE, null=True)
    ipoacusia = models.CharField(max_length=2, choices=LEFT_RIGHT_TYPE, null=True)
    acufeni = models.CharField(max_length=2, choices=LEFT_RIGHT_TYPE, null=True)
    sindr_ver = models.CharField(max_length=2, choices=SINDR_VER_TYPE, null=True)
    febbre = models.BooleanField(blank=True, default=False)
    uso_farmaci = models.BooleanField(blank=True, default=False)
    lacrimazione = models.BooleanField(blank=True, default=False)
    fotofobia = models.BooleanField(blank=True, default=False)
    prurito_cong = models.BooleanField(blank=True, default=False)
    bruciore_cong = models.BooleanField(blank=True, default=False)

    #Esame del medico
    PIR_NAS_TYPE = (
        ('NFM', 'normoformata'),
        ('GIB', 'gibbo'),
        ('SCO', 'scoiosi'),
        ('DEF', 'deformazioni varie')
    )
    VALV_NAS_TYPE = (
        ('NFN', 'normofunzionante'),
        ('INS', 'insufficienza sinistra'),
        ('IND', 'insufficienza destra'),
        ('INE', 'insufficienza bilaterale')
    )
    SETTO_NAS_TYPE = (
        ('ASS', 'in asse'),
        ('DVS', 'deviato a sinistra'),
        ('DVD', 'deviato a destra'),
        ('ESI', 'esse italica')
    )
    TURB_TYPE = (
        ('NTR', 'normotrofici'),
        ('IPT', 'ipertrofici'),
        ('IPE', 'iperemici'),
        ('EMA', 'ematosi')
    )

    pir_nas = models.CharField(max_length=3, choices=PIR_NAS_TYPE, null=True)
    setto_nas = models.CharField(max_length=3, choices=SETTO_NAS_TYPE, null=True)
    turb = models.CharField(max_length=3, choices=TURB_TYPE, null=True)
    polip_nas_sx = models.IntegerField(choices=list(zip(range(1, 5), range(1, 5))), unique=True, null=True)
    polip_nas_dx = models.IntegerField(choices=list(zip(range(1, 5), range(1, 5))), unique=True, null=True)
    essudato = models.CharField(max_length=2, choices=RINORREA_ESSUDATO_TYPE, null=True)
    ipertr_adenoidea = models.IntegerField(choices=list(zip(range(1, 5), range(1, 5))), unique=True, null=True)
    prick_test = models.BooleanField(blank=True, default=False)
    #allergy


class Diagnosis(ApiModel):
    #constants name of files
    FACTS_FILENAME = os.path.join(settings.PROJECT_PATH, 'fatti.clp')
    FUNCTIONS_FILENAME = os.path.join(settings.PROJECT_PATH, 'funztions.clp')
    DIAGNOSIS_FILENAME = os.path.join(settings.PROJECT_PATH, 'diagnosi.clp')

    #fields
    patient = models.ForeignKey(Patient, related_name='diagnosis', on_delete=models.CASCADE)
    cell_extraction = models.ForeignKey(CellExtraction, related_name='diagnosis', on_delete=models.CASCADE)

    def save(self, *args, **kwargs):
        import clips
        env = clips.Environment()
        env.load(FACTS_FILENAME)
        env.load(FUNCTIONS_FILENAME)
        env.load(DIAGNOSIS_FILENAME)
        super().save(*args, **kwargs);
