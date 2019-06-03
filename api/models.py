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
    other_num = models.SmallIntegerField(default=0)

    def get_cell_grade(self, cell_category):
        """
        Return the grade of a cell category in the cell extraction
        """
        if cell_category == "mucipara":
            return self.count300(self.mucipara_num)
        elif cell_category == "epithelium":
            return self.count300(self.epithelium_num)
        elif cell_category == "neutrophil":
            return self.count100(self.neutrophil_num)
        elif cell_category == "eosinophil":
            return self.count30(self.eosinophil_num)
        elif cell_category == "mastocyte":
            return self.count30(self.mastocyte_num)
        elif cell_category == "lymphocyte":
            return self.count30(self.lymphocyte_num)
        else:
            return 0

    @staticmethod
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


    def count100(self, cell_number):
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

    def count30(self, cell_number):
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

    def count16(self, cell_number):
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

        #end for
        #if patient has at least one anamnesis registered
        #if(Anamnesis.objects.filter(patient=self.patient).count() > 0):
            #calculate diagnosis
            #last_patient_anamnesis = Anamnesis.objects.filter(patient=self.patient).last()
            #create istance of diagnosis with the last anamnesis available
            #Diagnosis.objects.create(patient=self.patient, cell_extraction=self.cell_extraction, anamnesis=last_patient_anamnesis)
            #creare una istanza di diagnosis con tutti i campi e nel save del model diagnosi fare tutti i calcoli con l'ultima anamnesi

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
        cell_extraction = self.slide.cell_extraction
        #update the number of cells per type in cell_extraction reference
        if self.cell_category == CellCategory.objects.get(name="epithelium"):
            cell_extraction.epithelium_num += 1
        elif self.cell_category == CellCategory.objects.get(name="neutrophil"):
            cell_extraction.neutrophil_num += 1
        elif self.cell_category == CellCategory.objects.get(name="eosinophil"):
            cell_extraction.eosinophil_num += 1
        elif self.cell_category == CellCategory.objects.get(name="mastocyte"):
            cell_extraction.mastocyte_num += 1
        elif self.cell_category == CellCategory.objects.get(name="lymphocyte"):
            cell_extraction.lymphocyte_num += 1
        elif self.cell_category == CellCategory.objects.get(name="mucipara"):
            cell_extraction.mucipara_num += 1
        elif self.cell_category == CellCategory.objects.get(name="other"):
            cell_extraction.other_num += 1
        cell_extraction.save()
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
    patient = models.ForeignKey(Patient, related_name='anamnesis', on_delete=models.CASCADE)
    anamnesis_date = models.DateTimeField(auto_now_add=True)
    allergy_gen = models.CharField(max_length=2, choices=ALLERG_TYPE, null=True)
    allergy_fra = models.CharField(max_length=2, choices=ALLERG_TYPE, null=True)
    polip_nas_gen = models.BooleanField(blank=True, default=False)
    polip_nas_fra = models.BooleanField(blank=True, default=False)
    asma_bronch_gen = models.BooleanField(blank=True, default=False)
    asma_bronch_fra = models.BooleanField(blank=True, default=False)
    appunti_anam_fam = models.TextField(null=True, max_length=120)

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
    rinorrea_espans = models.CharField(max_length=2, choices=LEFT_RIGHT_TYPE, null=True)
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
    appunti_sintom = models.TextField(null=True, max_length=120)

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

    #Esame medico
    pir_nas = models.CharField(max_length=3, choices=PIR_NAS_TYPE, null=True)
    valv_nas = models.CharField(max_length=3, choices=VALV_NAS_TYPE, null=True)
    setto_nas = models.CharField(max_length=3, choices=SETTO_NAS_TYPE, null=True)
    turb = models.CharField(max_length=3, choices=TURB_TYPE, null=True)
    polip_nas_sx = models.IntegerField(choices=list(zip(range(1, 5), range(1, 5))), unique=True, null=True)
    polip_nas_dx = models.IntegerField(choices=list(zip(range(1, 5), range(1, 5))), unique=True, null=True)
    essudato = models.CharField(max_length=2, choices=RINORREA_ESSUDATO_TYPE, null=True)
    ipertr_adenoidea = models.IntegerField(choices=list(zip(range(1, 5), range(1, 5))), unique=True, null=True)
    appunti_alter_rinofaringe = models.TextField(null=True, max_length=120)
    esame_orecchie = models.TextField(null=True, max_length=120)
    conclusioni_esame = models.TextField(null=True, max_length=120)

    def save(self, *args, **kwargs):
        #saves and create diagnosis with the last available extraction
        super().save(*args, **kwargs);
        last_extraction = CellExtraction.objects.filter(patient=self.patient).last()
        #DiagnosisExtraction.objects.create(patient=self.patient, cell_extraction=last_extraction, anamnesis=self)


class Allergy(ApiModel):
    type = models.CharField(max_length=25)

    def evaluate_month(type, month):
        import numpy as np
        mesi = np.zeros(36)
        #graminacee
        if type == "Graminacee":
            for n in range(7, 28):
                mesi[n] = 1
            for n in range(10, 20):
                mesi[n] = 2
            for n in range(12, 16):
                mesi[n] = 3
        #cupressacee
        elif type == "Cupressacee/Taxacee":
            for n in range(0, 12):
                mesi[n] = 1
            mesi[4] = 2
            mesi[9] = 2
            for n in range(5, 9):
                mesi[n] = 3
        #nocciolo
        elif type == "Nocciolo":
            for n in range(2, 11):
                mesi[n] = 1
            mesi[5] = 2
            mesi[6] = 2
        #ontano
        elif type == "Ontano":
            for n in range(2, 10):
                mesi[n] = 1
            mesi[5] = 2
            mesi[6] = 2
            for n in range(13, 17):
                mesi[n] = 1
        #pioppo
        elif type == "Pioppo":
            for n in range(5, 11):
                mesi[n] = 1
            mesi[7] = 3
            mesi[8] = 2
        #frassino
        elif type == "Frassino comune":
            for n in range(5, 13):
                mesi[n] = 1
            for n in range(8, 12):
                mesi[n] = 2
        #betulla
        elif type == "Betulla":
            for n in range(8, 11):
                mesi[n] = 3
            mesi[11] = 2
            mesi[7] = 2
            for n in range(12, 15):
                mesi[n] = 1
        #salice
        elif type == "Salice":
            for n in range(7, 12):
                mesi[n] = 1
        #carpino nero
        elif type == "Carpino nero":
            for n in range(7, 18):
                mesi[n] = 1
            mesi[19] = 1
            for n in range(8, 13):
                mesi[n] = 3
        #quercia
        elif type == "Quercia":
            for n in range(8, 15):
                mesi[n] = 1
            for n in range(9, 13):
                mesi[n] = 2
            for n in range(10, 12):
                mesi[n] = 3
        #poligonacee
        elif type == "Poligonacee":
            for n in range(9, 16):
                mesi[n] = 1
        #orniello
        elif type == "Orniello":
            for n in range(9, 18):
                mesi[n] = 1
            for n in range(10, 15):
                mesi[n] = 2
            for n in range(11, 14):
                mesi[n] = 3
        #urticacee
        elif type == "Urticacee":
            for n in range(9, 35):
                mesi[n] = 1
            for n in range(11, 28):
                mesi[n] = 2
            for n in range(16, 25):
                mesi[n] = 3
        #castagno
        elif type == "Castagno":
            for n in range(14, 21):
                mesi[n] = 1
            mesi[16] = 2
            mesi[17] = 3
            mesi[18] = 2
        #platano
        elif type == "Platano":
            mesi[8] = 2
            mesi[10] = 2
            mesi[11] = 1
            mesi[9] = 3
        #pinacee
        elif type == "Pinacee":
            for n in range(9, 21):
                mesi[n] = 1
            for n in range(25, 35):
                mesi[n] = 1
            for n in range(11, 17):
                mesi[n] = 2
            for n in range(12, 15):
                mesi[n] = 3
            for n in range(26, 29):
                mesi[n] = 2
            mesi[30] = 2
        #piantaggine
        elif type == "Piantaggine":
            for n in range(9, 28):
                mesi[n] = 1
            for n in range(12, 26):
                mesi[n] = 2
            for n in range(14, 24):
                mesi[n] = 3
        #assenzio
        elif type == "Urticacee":
            for n in range(19, 23):
                mesi[n] = 1
            mesi[22] = 2
            mesi[26] = 2

        finemese = mesi[month * 3 - 1]
        metamese = mesi[month * 3 - 2]
        iniziomese = mesi[month * 3 - 3]
        media = (finemese + metamese + iniziomese)/3
        return media


class PrickTest(ApiModel):
    anamnesis = models.ForeignKey(Anamnesis, related_name='prick_test', on_delete=models.CASCADE)
    allergy = models.ForeignKey(Allergy, related_name='prick_test', on_delete=models.CASCADE)
    period = models.CharField(max_length=15, null=True)
    date = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        #assert facts for diagnosis calculation
        ##TO-DO prendere il mese dalla data!!!!!
        self.period = self.get_presence(Allergy.evaluate_month(self.allergy.type, self.anamnesis.anamnesis_date.month))
        #########
        super().save(*args, **kwargs);

    def get_presence(self, media):
        if media >= 0 and media < 1.50:
            return "Apollinico"
        if media >= 1.50 and media < 3.50:
            return "Pollinico"

class DiagnosisExtraction(ApiModel):
    #constants name of files
    FACTS_FILENAME = os.path.join(settings.PROJECT_PATH, 'fatti.clp')
    FUNCTIONS_FILENAME = os.path.join(settings.PROJECT_PATH, 'functions.clp')
    DIAGNOSIS_FILENAME = os.path.join(settings.PROJECT_PATH, 'diagnosi.clp')

    #fields
    patient = models.ForeignKey(Patient, related_name='diagnosis_extraction', on_delete=models.CASCADE)
    cell_extraction = models.ForeignKey(CellExtraction, related_name='diagnosis_extraction', on_delete=models.CASCADE)
    anamnesis = models.ForeignKey(Anamnesis, related_name='diagnosis_extraction', on_delete=models.CASCADE)
    diagnosis_date = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        #control if diagnosis has already been made with the combination
        #of cell extraction and anamnesis ids
        if DiagnosisExtraction.objects.filter(cell_extraction=self.cell_extraction, anamnesis=self.anamnesis).count() > 0:
            raise Exception('The diagnosis has already been made')
        #assert facts for diagnosis calculation
        super().save(*args, **kwargs);
        self.assert_facts()

    def assert_facts(self):
        import clips
        #initialize clips Environment
        env = clips.Environment()

        #load clips files
        env.load(self.FACTS_FILENAME)
        env.load(self.FUNCTIONS_FILENAME)
        env.load(self.DIAGNOSIS_FILENAME)

        #defining fact templates
        cellula_template = env.find_template("cellula")
        famiglia_template = env.find_template("famiglia")
        sintomo_template = env.find_template("sintomo")
        scoperta_template = env.find_template("scoperta")
        rinomanometria_template = env.find_template("rinomanometria")
        diagnosi_template = env.find_template("diagnosi")
        prick_test_template = env.find_template("prick-test")

        #assert facts cellule
        for cell_category in CellCategory.objects.all():
            cellula_fact = cellula_template.new_fact()
            cellula_fact['nome'] = clips.Symbol(cell_category.name.capitalize())
            cellula_fact['grado'] = self.cell_extraction.get_cell_grade(cell_category.name)
            cellula_fact.assertit()

        if self.anamnesis.allergy_gen != None:
            famiglia_fact = famiglia_template.new_fact()
            famiglia_fact['soggetto'] = clips.Symbol("genitore")
            famiglia_fact['disturbo'] = clips.Symbol("allergia")
            famiglia_fact['tipo'] = clips.Symbol(self.anamnesis.get_allergy_gen_display())
            famiglia_fact.assertit()

        if self.anamnesis.allergy_fra != None:
            famiglia_fact = famiglia_template.new_fact()
            famiglia_fact['soggetto'] = clips.Symbol("fratello")
            famiglia_fact['disturbo'] = clips.Symbol("allergia")
            famiglia_fact['tipo'] = clips.Symbol(self.anamnesis.get_allergy_fra_display())
            famiglia_fact.assertit()

        if self.anamnesis.polip_nas_gen == True:
            famiglia_fact = famiglia_template.new_fact()
            famiglia_fact['soggetto'] = clips.Symbol("genitore")
            famiglia_fact['disturbo'] = clips.Symbol("poliposi")
            famiglia_fact.assertit()

        if self.anamnesis.polip_nas_fra == True:
            famiglia_fact = famiglia_template.new_fact()
            famiglia_fact['soggetto'] = clips.Symbol("fratello")
            famiglia_fact['disturbo'] = clips.Symbol("poliposi")
            famiglia_fact.assertit()

        if self.anamnesis.asma_bronch_gen == True:
            famiglia_fact = famiglia_template.new_fact()
            famiglia_fact['soggetto'] = clips.Symbol("genitore")
            famiglia_fact['disturbo'] = clips.Symbol("asma")
            famiglia_fact.assertit()

        if self.anamnesis.asma_bronch_fra == True:
            famiglia_fact = famiglia_template.new_fact()
            famiglia_fact['soggetto'] = clips.Symbol("fratello")
            famiglia_fact['disturbo'] = clips.Symbol("asma")
            famiglia_fact.assertit()

        if self.anamnesis.ostr_nas != None:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Ostruzione nasale " + self.anamnesis.get_ostr_nas_display())
            sintomo_fact.assertit()

        if self.anamnesis.rinorrea != None:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Rinorrea nasale " + self.anamnesis.get_rinorrea_display())
            sintomo_fact.assertit()
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Espansione rinorrea: " + self.anamnesis.get_rinorrea_espans_display())
            sintomo_fact.assertit()

        if self.anamnesis.prur_nas == True:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Prurito nasale")
            sintomo_fact.assertit()

        if self.anamnesis.starnutazione != None:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Starnutazione " + self.anamnesis.get_starnutazione_display())
            sintomo_fact.assertit()

        if self.anamnesis.prob_olf == True:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Problemi olfattivi dovuti a " + self.anamnesis.get_prob_olf_display())
            sintomo_fact.assertit()

        if self.anamnesis.ovatt_aur != None:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Ovattamento " + self.anamnesis.get_ovatt_aur_display())
            sintomo_fact.assertit()

        if self.anamnesis.ipoacusia != None:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Ipoacusi " + self.anamnesis.get_ipoacusia_display())
            sintomo_fact.assertit()

        if self.anamnesis.acufeni != None:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Acufeni " + self.anamnesis.get_acufeni_display())
            sintomo_fact.assertit()

        if self.anamnesis.sindr_ver != None:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Sindrome vertiginosa " + self.anamnesis.get_sindr_ver_display())
            sintomo_fact.assertit()

        if self.anamnesis.febbre == True:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Febbre")
            sintomo_fact.assertit()

        if self.anamnesis.uso_farmaci == True:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Uso eccessivo di farmaci")
            sintomo_fact.assertit()

        if self.anamnesis.lacrimazione == True:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Lacrimazione")
            sintomo_fact.assertit()

        if self.anamnesis.fotofobia == True:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Fotofobia")
            sintomo_fact.assertit()

        if self.anamnesis.prurito_cong == True:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Prurito occhio")
            sintomo_fact.assertit()

        if self.anamnesis.bruciore_cong == True:
            sintomo_fact = sintomo_template.new_fact()
            sintomo_fact['nome'] = clips.Symbol("Bruciore")
            sintomo_fact.assertit()

        if self.anamnesis.pir_nas != None and self.anamnesis.get_pir_nas_display() != 'normoformata':
            scoperta_fact = scoperta_template.new_fact()
            scoperta_fact['parte-anatomica'] = clips.Symbol("piramide-nasale")
            scoperta_fact['caratteristica'] = clips.Symbol(self.anamnesis.get_pir_nas_display())
            scoperta_fact.assertit()

        if self.anamnesis.valv_nas != None and self.anamnesis.get_valv_nas_display() != 'normofunzionante':
            scoperta_fact = scoperta_template.new_fact()
            scoperta_fact['parte-anatomica'] = clips.Symbol("valvola-nasale")
            scoperta_fact['caratteristica'] = clips.Symbol(self.anamnesis.get_valv_nas_display())
            scoperta_fact.assertit()

        if self.anamnesis.setto_nas != None and self.anamnesis.get_setto_nas_display() != 'in asse':
            scoperta_fact = scoperta_template.new_fact()
            scoperta_fact['parte-anatomica'] = clips.Symbol("setto-nasale")
            scoperta_fact['caratteristica'] = clips.Symbol(self.anamnesis.get_setto_nas_display())
            scoperta_fact.assertit()

        if self.anamnesis.turb != None and self.anamnesis.get_turb_display() != 'normoformata':
            scoperta_fact = scoperta_template.new_fact()
            scoperta_fact['parte-anatomica'] = clips.Symbol("turbinati")
            scoperta_fact['caratteristica'] = clips.Symbol(self.anamnesis.get_turb_display())
            scoperta_fact.assertit()

        if self.anamnesis.polip_nas_sx != None:
            scoperta_fact = scoperta_template.new_fact()
            scoperta_fact['parte-anatomica'] = clips.Symbol("poliposi-sinistra")
            scoperta_fact['caratteristica'] = clips.Symbol(self.anamnesis.polip_nas_sx)
            scoperta_fact.assertit()

        if self.anamnesis.polip_nas_dx != None:
            scoperta_fact = scoperta_template.new_fact()
            scoperta_fact['parte-anatomica'] = clips.Symbol("poliposi-destra")
            scoperta_fact['caratteristica'] = clips.Symbol(self.anamnesis.polip_nas_dx)
            scoperta_fact.assertit()

        if self.anamnesis.essudato != None:
            scoperta_fact = scoperta_template.new_fact()
            scoperta_fact['parte-anatomica'] = clips.Symbol("essudato")
            scoperta_fact['caratteristica'] = clips.Symbol(self.anamnesis.get_essudato_display())
            scoperta_fact.assertit()

        #allergy assert
        for prick_test in PrickTest.objects.filter(anamnesis=self.anamnesis):
            prick_test_fact = prick_test_template.new_fact()
            prick_test_fact["esito"] = clips.Symbol("positivo")
            prick_test_fact["periodo"] = clips.Symbol(prick_test.period)
            prick_test_fact["allergene"] = clips.Symbol(prick_test.allergy.type)
            prick_test_fact.assertit()
        #run the diagnosis calculation
        env.run()
        #evalutate all diagnosis
        extracted_diagnosis = env.eval("(find-all-facts((?f diagnosi)) TRUE)")

        for fact in env.facts():
            print(fact)

        #loop through diagnosis extracted from run
        for diagnosis_extr in extracted_diagnosis:
            #create instance of diagnosis with name and information
            Diagnosis.objects.create(diagnosis_extraction=self, name=diagnosis_extr["nome"], information=diagnosis_extr["informazioni"])
        #clear the environment
        env.clear()

class Diagnosis(ApiModel):
    #fields
    diagnosis_extraction = models.ForeignKey(DiagnosisExtraction, related_name='diagnosis', on_delete=models.CASCADE)
    name = models.CharField(max_length=30)
    information = models.CharField(max_length=300)

class Report(ApiModel):
    #fields
    report_date = models.DateTimeField(auto_now_add=True)
    anamnesis = models.ForeignKey(Anamnesis, related_name='report', null=True, on_delete=models.CASCADE)
    cell_extraction = models.ForeignKey(CellExtraction, related_name='report', null=True, on_delete=models.CASCADE)
    report_file = models.FileField(upload_to='report/', null=True)

    # the media folder name
    MEDIA_FOLDER = 'report'


    def save(self, *args, **kwargs):
        #control if report has already been made with the combination
        #of cell extraction and anamnesis ids

        self.createReport()
        super().save(*args, **kwargs)

    def createReport(self):
        from fpdf import FPDF
        # Letter size paper, use inches as unit of measure
        pdf = FPDF(format='letter', unit='in')


        # Add new page. Without this you cannot create the document.
        pdf.add_page()

        # Remember to always put one of these at least once.
        pdf.set_font('Times', '', 14.0)

        # Effective page width, or just epw
        epw = pdf.w - 2 * pdf.l_margin

        # Set column width to 1/4 of effective page width to distribute content
        # evenly across table and page
        col_width = epw / 4

        # Text height is the same as current font size
        th = pdf.font_size

        if(self.cell_extraction != None):
            title = ['Nome', 'Conta cellule', 'Grado']
            data = [['Neutrofili',  self.cell_extraction.neutrophil_num,  self.cell_extraction.get_cell_grade('neutrophil')],
                    ['Epiteliali',  self.cell_extraction.epithelium_num,  self.cell_extraction.get_cell_grade('epithelium')],
                    ['Linfociti',   self.cell_extraction.lymphocyte_num,  self.cell_extraction.get_cell_grade('lymphocyte')],
                    ['Mucipare',    self.cell_extraction.mucipara_num,    self.cell_extraction.get_cell_grade('mucipara')],
                    ['Eosinofili',  self.cell_extraction.lymphocyte_num,  self.cell_extraction.get_cell_grade('eosinophil')],
                    ['Mastcellule', self.cell_extraction.mastocyte_num,   self.cell_extraction.get_cell_grade('mastocyte')],
                    ['Altro',       self.cell_extraction.other_num,       "-"],
                    ]

            # Document title centered, 'B'old, 14 pt
            pdf.set_font('Times', 'B', 20.0)
            pdf.cell(epw, 0.0, 'Report', align='C')
            pdf.set_font('Times', '', 14.0)
            pdf.ln(0.5)



            # Line break equivalent to 4 lines
            pdf.ln(4 * th)

            pdf.set_fill_color( 100, 100, 100)
            for row in title:
                pdf.cell(col_width, th, row, border=1 ,fill=True)

            pdf.ln(th)
            pdf.set_fill_color( 255, 255, 255)
            for row in data:
                for datum in row:
                    # Use the function str to coerce any input to string type.
                    pdf.cell(col_width, th, str(datum), border=1)
                    # pdf.cell(col_width, 2 * th, str(datum), border=1) per spaziare di piu

                pdf.ln(th)
        pdf.ln()
        pdf.ln()


        if(self.anamnesis != None):
            pdf.set_font('Times', 'B', 20.0)
            pdf.cell(epw, 0.0, 'Anamnesi', align='C')
            pdf.set_font('Times', '', 14.0)
            pdf.ln(0.5)
            pdf.cell(col_width, th, "Anamnesi familiare")
            pdf.ln()

            pdf.set_font('Times', '', 12.0)

            pdf.multi_cell(col_width, th, "Allergia Genitori: " + Report.getYesOrNo(self.anamnesis.get_allergy_gen_display()))
            pdf.cell(col_width, th, "Tipo Allergia Genitori: " + Report.ifNotNull(self.anamnesis.get_allergy_gen_display()))
            pdf.ln()
            pdf.multi_cell(col_width, th, "Allergia Fratelli: " + Report.getYesOrNo(self.anamnesis.get_allergy_fra_display()))
            pdf.cell(col_width, th, "Tipo Allergia Fratelli: " + Report.ifNotNull(self.anamnesis.get_allergy_fra_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Poliposi Genitori: " + Report.getBooleanValue(self.anamnesis.polip_nas_gen))
            pdf.ln()
            pdf.cell(col_width, th, "Asma Genitori: " + Report.getBooleanValue(self.anamnesis.asma_bronch_gen))
            pdf.ln()
            pdf.cell(col_width, th, "Asma Fratelli: " + Report.getBooleanValue(self.anamnesis.asma_bronch_fra))
            pdf.ln()
            pdf.cell(col_width, th, "Appunti Anamnesi Familiare:")
            pdf.ln()
            pdf.ln()

            pdf.cell(col_width, th, "Patologica prossima - Sintomi")
            pdf.ln()
            pdf.cell(col_width, th, "Ostruzione: " + Report.ifNotNull(self.anamnesis.get_ostr_nas_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Rinorrea: " + Report.ifNotNull(self.anamnesis.get_rinorrea_display()) + " " + Report.ifNotNull(self.anamnesis.get_rinorrea_espans_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Prurito nasale: " + Report.getBooleanValue(self.anamnesis.prur_nas))
            pdf.ln()
            pdf.cell(col_width, th, "Starnutazione: " + Report.ifNotNull(self.anamnesis.get_starnutazione_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Problemi olfattivi: " + Report.ifNotNull(self.anamnesis.get_prob_olf_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Ovattamento auricolare: " + Report.ifNotNull(self.anamnesis.get_ovatt_aur_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Ipoacusia: " + Report.ifNotNull(self.anamnesis.get_ipoacusia_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Acufeni: " + Report.ifNotNull(self.anamnesis.get_acufeni_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Sindrome vertiginosa: " + Report.ifNotNull(self.anamnesis.get_sindr_ver_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Febbre: " + Report.getBooleanValue(self.anamnesis.febbre))
            pdf.ln()
            pdf.cell(col_width, th, "Uso farmaci: " + Report.getBooleanValue(self.anamnesis.uso_farmaci))
            pdf.ln()
            pdf.cell(col_width, th, "Lacrimazione: " + Report.getBooleanValue(self.anamnesis.lacrimazione))
            pdf.ln()
            pdf.cell(col_width, th, "Fotofobia: " + Report.getBooleanValue(self.anamnesis.fotofobia))
            pdf.ln()
            pdf.cell(col_width, th, "Prurito : " + Report.getBooleanValue(self.anamnesis.prurito_cong))
            pdf.ln()
            pdf.cell(col_width, th, "Bruciore: " + Report.getBooleanValue(self.anamnesis.bruciore_cong))
            pdf.ln()
            pdf.cell(col_width, th, "Appunti Anamnesi Patologica Prossima:")
            pdf.ln()
            pdf.ln()
            pdf.cell(col_width, th, "Esame obbiettivo strumentale del naso")
            pdf.ln()
            pdf.cell(col_width, th, "Piramide nasale: " + Report.ifNotNull(self.anamnesis.get_pir_nas_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Valvola nasale: " + Report.ifNotNull(self.anamnesis.get_valv_nas_display()))
            pdf.ln()
            pdf.ln()
            pdf.cell(col_width, th, "Endoscopia nasale, Esame Rinomanometrico, Esame Otoscopico e Allergologico")
            pdf.ln()
            pdf.cell(col_width, th, "Setto nasale: " + Report.ifNotNull(self.anamnesis.get_setto_nas_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Turbinati: " + Report.ifNotNull(self.anamnesis.get_turb_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Poliposi lato sinistro: " + str(self.anamnesis.polip_nas_sx))
            pdf.ln()
            pdf.cell(col_width, th, "Poliposi lato destro: " + str(self.anamnesis.polip_nas_dx))
            pdf.ln()
            pdf.cell(col_width, th, "Essudato: " + Report.ifNotNull(self.anamnesis.get_essudato_display()))
            pdf.ln()
            pdf.cell(col_width, th, "Ipertrofia: " + str(self.anamnesis.ipertr_adenoidea))
            pdf.ln()
            pdf.cell(col_width, th, "Appunti Esame Rinomanometrico:")
            pdf.ln()
            pdf.cell(col_width, th, "Allergia/e: " + self.getAllergy())
            pdf.ln()
            pdf.cell(col_width, th, "Base Dx:")
            pdf.ln()
            pdf.cell(col_width, th, "Base Sx e Dx:")
            pdf.ln()
            pdf.cell(col_width, th, "Decongestione Sx:")
            pdf.ln()
            pdf.cell(col_width, th, "Decongestione Dx:")
            pdf.ln()
            pdf.cell(col_width, th, "Decongestione Sx e Dx")
            pdf.ln()

        report_path = Report.getMediaPath()

        if self.anamnesis != None and self.cell_extraction != None:
            report_filename = Report.generateFilename(self.anamnesis.id, self.cell_extraction.id)
        elif self.anamnesis == None:
            report_filename = Report.generateFilename(0, self.cell_extraction.id)
        elif self.cell_extraction == None:
            report_filename = Report.generateFilename(self.anamnesis.id, 0)


        # create the report path if doesn't exists
        if not os.path.isdir(report_path):
            os.mkdir(report_path)

        report_path = os.path.join(report_path, report_filename)
        pdf.output(name=report_path, dest='F').encode('latin-1')
        #update the file field of the report
        self.report_file = Report.getReportRelativePath(report_filename)

    @staticmethod
    def getMediaPath():
        return os.path.join(settings.MEDIA_ROOT, Report.MEDIA_FOLDER)

    @staticmethod
    def getReportRelativePath(filename):
        return os.path.join(Report.MEDIA_FOLDER, filename)

    @staticmethod
    def generateFilename(anamnesis_id, cell_extraction_id):
        return "Report" + str(anamnesis_id) + str(cell_extraction_id) + ".pdf"

    #control if field is null
    @staticmethod
    def ifNotNull(field):
        if(field != None):
            return field
        return "_____"

    @staticmethod
    def getYesOrNo(field):
        if(field != None):
            return "si"
        return "no"

    @staticmethod
    def getBooleanValue(field):
        if(field == True):
            return "si"
        return "no"

    #return all allergies in a string
    def getAllergy(self):
        allergies = "Non presenti"
        #if there are allergies
        if(PrickTest.objects.filter(anamnesis=self.anamnesis).count() > 0):
            allergies = ""
            for prick_test in PrickTest.objects.filter(anamnesis=self.anamnesis):
                allergies += (prick_test.allergy.type + ", ")
        return allergies
