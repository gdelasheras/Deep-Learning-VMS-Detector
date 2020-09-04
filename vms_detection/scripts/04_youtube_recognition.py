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

from utils import configuration as conf
from utils import functions
from argparse import ArgumentParser
from keras_retinanet import models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pafy

if __name__ == "__main__":
    
    # Arguments.
    parser = ArgumentParser()

    parser.add_argument("-u", "--url", 
        dest="url",
        help="url",
        type= str,
        required=True)
    
    parser.add_argument("-b", "--backbone_name", 
        dest="backbone_name",
        help="backbone_name",
        type= str,
        required=True)
    
    parser.add_argument("-d", "--destiny", 
        dest="destiny",
        help="destiny",
        type= str,
        required=True)

    args = parser.parse_args()
    url = args.url
    backbone_name = args.backbone_name
    destiny = args.destiny
    
    recognized = 1
    
    # Model load.
    model = functions.load_model(backbone_name, conf.PATH_SNAPSHOTS + 'resnet50_color/')
    
    if model is None:
        print("Model not valid!")
        quit()
    
    # Destiny folder.
    if destiny != "autolabeling" and destiny != "test":
        print("Destiny not valid!")
        quit()
    elif destiny == "autolabeling":
        destiny = conf.PATH_AUTOLABELING
    elif destiny == "test":
        destiny = conf.PATH_TEST
        
    print("Saving in: " + destiny)
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    vPafy = pafy.new(url)
    play = vPafy.getbest()

    # Video.
    cap = cv2.VideoCapture(play.url)
    print(str(recognized - 1) + " VMS found...")
    
    last_frame = 0

    while(cap.isOpened()):
        
        # Next Frame.
        ret, frame = cap.read()
        
        if ret:
            # Recognition.
            if last_frame == 0:
                img = cv2.resize(frame, (1024, 768))
                boxes, scores, _ = functions.predict(img, model)

                # Threshold.
                if scores[0][0] >= conf.MIN_THRESHOLD:
                    cv2.imwrite(destiny + url.split("/")[-1] + "_" + str(recognized) + ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    recognized += 1
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(str(recognized - 1) + " VMS found...")
                    last_frame = 30
            else:
                last_frame -= 1 

        else:
            break
    
    print("FINISHED!")