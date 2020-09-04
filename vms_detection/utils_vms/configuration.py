######################################################################################################################################################
#
# Gonzalo de las Heras de Matías. July 2020.
#
# Universidad Europea de Madrid. Business & Tech School. IBM Master's Degree in Big Data Analytics.
#
# Master's thesis:
#
# ADVANCED DRIVER ASSISTANCE SYSTEM (ADAS) BASED ON AUTOMATIC LEARNING TECHNIQUES FOR THE DETECTION AND TRANSCRIPTION OF VARIABLE MESSAGE SIGNS.
#
######################################################################################################################################################

# Files
FILE_TRAIN_ANNOTATIONS = 'annotations_train.csv'
FILE_VALIDATION_ANNOTATIONS = 'annotations_validation.csv'
FILE_TEST_ANNOTATIONS = 'annotations_test.csv'
FILE_CLASSES = 'classes.csv'
FILE_DATA = 'data.csv'

# Paths
PATH_SNAPSHOTS = 'snapshots/'
PATH_SNAPSHOTS_PRETRAINED = PATH_SNAPSHOTS + "pretrained/"
PATH_SNAPSHOTS_RESNET50 = PATH_SNAPSHOTS + "resnet50/"
PATH_SNAPSHOTS_RESNET101 = PATH_SNAPSHOTS + "resnet101/"
PATH_SNAPSHOTS_RESNET152 = PATH_SNAPSHOTS + "resnet152/"
PATH_DATASET = '../dataset/'
PATH_IMAGES = PATH_DATASET + 'images/'
PATH_ORIGINAL = PATH_IMAGES + '00_original/'
PATH_CLEAN = PATH_IMAGES + '01_clean/'
PATH_RESIZED = PATH_IMAGES + '02_resized/'
PATH_AUTOLABELING = PATH_IMAGES + '04_autolabeling/'
PATH_TEST = PATH_IMAGES + '05_test/'
PATH_ANNOTATIONS = 'annotations/'

# Folder names
FOLDER_SNAPSHOTS = 'snapshots'
FOLDER_DATASET = 'dataset'

# Pretrained models
PRETRAINED_MODELS_BASE_URL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/'

PRETRAINED_RESNET50   = { 'repo_name': 'resnet50_coco_best_v2.1.0.h5', 
                          'snapshot_name': 'pretrained_resnet50.h5', 
                          'backbone_name': 'resnet50'  
                        }

PRETRAINED_RESNET101  = { 'repo_name': 'resnet101_oid_v1.0.0.h5',
                          'snapshot_name': 'pretrained_resnet101.h5',
                          'backbone_name': 'resnet101' 
                        }

PRETRAINED_RESNET152  = { 'repo_name': 'resnet152_oid_v1.0.0.h5',
                          'snapshot_name': 'pretrained_resnet152.h5',
                          'backbone_name': 'resnet152' 
                        }

# Others
RANDOM_SEED = 42
MIN_THRESHOLD = 0.94