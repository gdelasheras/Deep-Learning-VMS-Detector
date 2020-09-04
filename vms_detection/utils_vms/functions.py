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

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet import models
from IPython.display import clear_output
from utils_vms import configuration as conf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
    
def predict(image, model):
    """
    This function predicts object within the given image with the given model.
    
    @param image: Image to predict.
    @param model: Model for predicting.
    
    return:
        - boxes: List of localizations (boxes).
        - scores: List of scores.
        - labels: List of labels.
    """
    
    # Prepocessing.
    image = preprocess_image(image)

    # Resize.
    image, _ = resize_image(image)
    
    # Predict.
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    return boxes, scores, labels

def download_pretrained_models():
    """
    This function downloads all resnet pretrained models.
    """
    
    # Pretrained models.
    pretrained_models = [conf.PRETRAINED_RESNET50, 
                         conf.PRETRAINED_RESNET101,
                         conf.PRETRAINED_RESNET152]
    
    for model in pretrained_models:
        if os.path.isfile(conf.PATH_SNAPSHOTS_PRETRAINED + model['snapshot_name']):
            # Already downloaded.
            print('Pretrained ' + model['snapshot_name'] + ' already exists')
        else:
            # Download.
            print('Downloading ' + model['repo_name'] + ' model...')
            urllib.request.urlretrieve(conf.PRETRAINED_MODELS_BASE_URL + model['repo_name'], 
                                       conf.PATH_SNAPSHOTS_PRETRAINED + model['snapshot_name'])
            print(model['repo_name'] + ' downloaded in ' + conf.PATH_SNAPSHOTS_PRETRAINED + "/" + model['snapshot_name'] + '.')

def load_model(backbone_name, path):
    """
    This function loads a resnet model (resnet50, resnet101, resnet152).
    
    @param backbone_name: Resnet name.
    
    return: The selected model.
    """
    
    # Backbone check.
    if backbone_name not in ['resnet50', 'resnet101', 'resnet152']:
        print("Model not found!")
        return None
    
    # Model load and conversion.
    print("Loading " + backbone_name)
    model_path = os.path.join(path, 
                              sorted(os.listdir(path), 
                              reverse=True)[0])
    model = models.load_model(model_path, backbone_name=backbone_name)
    model = models.convert_model(model)
    
    # Clear output.
    os.system('cls' if os.name == 'nt' else 'clear')
    clear_output()
    
    return model