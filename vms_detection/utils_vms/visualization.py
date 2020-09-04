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
from utils_vms import functions
from utils_vms import configuration as conf
import matplotlib.pyplot as plt
import random
import cv2
import os

def show_examples(df, N=8):
    """
    This function shows N random images from df.
    
    @param df: Dataset.
    """
    
    # Random 8 samples.
    examples = random.sample(range(0, len(df.index)), N)
    
    # Plot configuration.
    fig, axs = plt.subplots(int(N/4), 4, figsize=(20, int(N/5) * 10 / 2))
    
    for _ in axs:
        for ax in _:
            
            # Example extraction.
            example = examples.pop()
            
            # Image read.
            img = cv2.cvtColor(read_image_bgr(df.iloc[example].image_name), cv2.COLOR_BGR2RGB)  
            
            if df.iloc[example].x_min != '':
                # Annotated box.
                box = [df.iloc[example].x_min, df.iloc[example].y_min, df.iloc[example].x_max, df.iloc[example].y_max]

                # Box draw.
                draw_box(img, box, color=(255, 0, 255), thickness=4)
            
            # Image show.
            ax.imshow(img)
            ax.set_title(df.iloc[example].image_name.split("/")[-1])
            ax.axis('off')

    plt.show()

def show_predicted(df, model_path, N=8, annotated=True):
    """
    This function shows N random images from df.
    
    @param df: Dataset.
    """
    
    # Random 8 samples.
    examples = random.sample(range(0, len(df.index)), N)
    
    # Plot configuration.
    fig, axs = plt.subplots(int(N/4), 4, figsize=(20, int(N/5) * 10 / 2))
    
    # Model load.
    model = functions.load_model(conf.PRETRAINED_RESNET50['backbone_name'], model_path)
    
    for _ in axs:
        for ax in _:
            
            # Example extraction.
            example = examples.pop()
            
            # Image read.
            img = cv2.cvtColor(read_image_bgr(df.iloc[example].image_name), cv2.COLOR_BGR2RGB)
            
            # Image preprocessing.
            img = cv2.resize(img, (1333, 800))
            
            # Prediction.
            pred_boxes, scores, labels = functions.predict(img, model)
            
            if annotated and df.iloc[example].x_min != '':
                
                #Annotated box.
                annotated_box = [df.iloc[example].x_min, df.iloc[example].y_min, df.iloc[example].x_max, df.iloc[example].y_max]
            
                # Box draw.
                draw_box(img, annotated_box, color=(255, 0, 255), thickness=4)

            # Predicted box draw.
            draw_detections(img, pred_boxes, scores, labels, color=(255, 255, 0))
            
            # Image show.
            ax.imshow(img)
            ax.set_title("VMS: " + str(scores[0][0] * 100) + "%")
            ax.axis('off')
    
    # Clear output.
    os.system('cls' if os.name == 'nt' else 'clear')
    
    plt.savefig('examples_vms.jpg', format='jpeg')
    
    print()
    plt.show()
    print()

    
def show_image_objects(image_row):
    """
    This function shows an image from a dataset.
    
    @param row: Dataset row.
    """

    # Image read.
    image = read_image_bgr(image_row.image_name)
    
    # Annotated box.
    box = [image_row.x_min, image_row.y_min, image_row.x_max, image_row.y_max]
    
    # Image copy.
    draw = image.copy()
    
    # Box draw.
    draw_box(draw, box, color=(255, 255, 0))

    # Image show.
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
    
def draw_detections(image, boxes, scores, labels, color=None):
    """
    This function draws the given boxes, scores, labels in a given image.
    
    @param image:
    @param boxes: List of localizations (boxes).
    @param scores: List of scores.
    @param labels: List of labels.
    """
    
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        
        # Min threshold.
        if score < 0:
            break
        
        # Color.
        if color is None:
            color = label_color(label)
        
        # Box draw.
        draw_box(image, box, color=color, thickness=4)
        
        # Label draw.
        draw_caption(image, box, "{} {:.3f}".format('VMS', score))
        
def show_detected_objects(image_row):
    """
    This function shows a given image with the predicted and the annotated box.
    
    @param image_row: Dataset row.
    """
    
    # Image read.
    image = read_image_bgr(image_row.image_name)

    # Prediction.
    boxes, scores, labels = predict(image)
    
    # Image copy.
    draw = image.copy()

    # Annotated box.
    true_box = [image_row.x_min, image_row.y_min, image_row.x_max, image_row.y_max]
    
    # True box draw.
    draw_box(draw, true_box, color=(255, 255, 0))

    # Predicted box draw.
    draw_detections(draw, boxes, scores, labels)

    # Image show.
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
    
def detect_image(image, model):
    """
    This function shows a given image with the predicted box.
    
    @param image: Image to show.
    @param model: Model used to predict.
    
    @return:
        - boxes: List of localizations (boxes).
        - scores: List of scores.
    """
    
    # Prediction.
    boxes, scores, labels = predict(image, model)

    # Image copy.
    draw = image.copy()

    # Predicted box draw.
    draw_detections(draw, boxes, scores, labels)

    # Image show.
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
    
    return boxes, scores

def show_image_box(image, box):
    """
    This function shows a given image with a given box.
    
    @param image: Image to show.
    @param box: Box to draw.
    """
    
    # Image copy.
    draw = image.copy()

    # Box draw.
    draw_box(image.copy(), box, color=(255, 0, 255), thickness=5)

    # Image show.
    plt.axis('off')
    plt.imshow(draw)
    plt.show()