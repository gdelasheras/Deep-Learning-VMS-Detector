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
from keras_retinanet import models
from argparse import ArgumentParser
from PIL import Image
import pafy
import numpy as np
import pandas as pd
import cv2
import os
import glob

if __name__ == "__main__":
    
    # Arguments.
    parser = ArgumentParser()
    
    parser.add_argument("-b", "--backbone_name", 
        dest="backbone_name",
        help="backbone_name",
        type= str,
        required=True)
    
    parser.add_argument("-v", "--vms", 
        dest="vms",
        help="vms",
        type= str,
        required=True)
    
    args = parser.parse_args()
    backbone_name = args.backbone_name
    vms = args.vms
    
    if vms != 'n' and vms != 's':
        print("Invalid no_vms!")
        quit()
        
    # Model load.
    model = functions.load_model(backbone_name, conf.PATH_SNAPSHOTS  + 'resnet50_color/')
    
    if model is None:
        print("Model not valid!")
        quit()
    
    # Images in folder.
    images = glob.glob(conf.PATH_AUTOLABELING + "*")

    # Dataset.
    data = pd.read_csv(conf.PATH_IMAGES + conf.FILE_DATA)
    files = data['original_name'].to_list()
    
    # Dataset next index.
    new_index = len(data.index)

    # Characters to scape.
    scape_chars = [(' ', '_'), ('@', ''), ('-', '_'), ('á', 'a'),
                   ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'),
                   ('(', ''), (')', ''), (',', '')
                  ]

    for path in images:
        
        # Prediction.
        img = cv2.imread(cv2.samples.findFile(path))
        img = cv2.resize(img, (1024, 768))
        
        if vms == 's':
            boxes, scores, _ = functions.predict(img, model)
        
            # Not enough confidence.
            if scores[0][0] <= 0.97:
                print("No VMS found!")
                continue
        
                # Detection box.
                boxes = boxes[0][0]

        # File name cleanning.
        img_name = path.split('/')[-1].lower()
        for char in scape_chars:
            img_name = img_name.replace(char[0], char[1])
        
        # Image already in the dataset.
        if img_name in files or path.split('/')[-1] in files:
            print("Already saved!")
            os.remove(path)
            continue

        # File reading.
        img = Image.open(path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Save in 00_original.
        img.save(conf.PATH_ORIGINAL + img_name, "JPEG", quality=80, optimize=True, progressive=True)

        # Save in 01_clean.
        img.save(conf.PATH_CLEAN + str(new_index).zfill(5) + ".jpg", quality=80, optimize=True, progressive=True)

        # Save in 02_resized.
        img = cv2.imread(cv2.samples.findFile(path))
        img = cv2.resize(img, (1920, 1080))
        cv2.imwrite(conf.PATH_RESIZED + 'imgs/' + str(new_index).zfill(5) + ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        
        if vms == 's':
            # Box resize.
            x_ratio = 1920 / 1024
            y_ratio = 1080 / 768
            boxes[0] = x_ratio * boxes[0]
            boxes[1] = y_ratio * boxes[1]
            boxes[2] = x_ratio * boxes[2]
            boxes[3] = y_ratio * boxes[3]
    
        # Dataset registration.
        data.loc[new_index, 'id'] = str(new_index)
        data.loc[new_index, 'original_name'] = img_name
        data.loc[new_index, 'name'] = str(new_index).zfill(5) + ".jpg"
        if vms == 's':
            data.loc[new_index, 'coordinates'] = str(int(boxes[0])) + ";" + str(int(boxes[1])) + ";" + str(int(boxes[2])) + ";" + str(int(boxes[3]))
        else:
            data.loc[new_index, 'coordinates'] = ";;;"
        data.loc[new_index, 'flipped_name'] = '-'
        data.loc[new_index, 'flipped_coordinates'] = '-'

        print('Image ' + str(new_index).zfill(5) + ' added!')
        
        # XML reading.
        xml = open(conf.PATH_IMAGES + "template.xml", "r")
        xml_content = xml.read()
        xml.close()

        if vms == 's':
            # XML annotations.
            xml_content = xml_content.replace('<xmin></xmin>', '<xmin>' + str(int(boxes[0])) + '</xmin>')
            xml_content = xml_content.replace('<ymin></ymin>', '<ymin>' + str(int(boxes[1])) + '</ymin>')
            xml_content = xml_content.replace('<xmax></xmax>', '<xmax>' + str(int(boxes[2])) + '</xmax>')
            xml_content = xml_content.replace('<ymax></ymax>', '<ymax>' + str(int(boxes[3])) + '</ymax>')
        
        xml_content = xml_content.replace('<filename></filename>', '<filename>' + str(new_index).zfill(5) + ".jpg" + '</filename>')
        xml_content = xml_content.replace('<path></path>', '<path>' + str(new_index).zfill(5) + ".jpg" + '</path>')
        
        # XML save.
        xml_annotations = open(conf.PATH_RESIZED + "coordinates/" + str(new_index).zfill(5) + ".xml", "x")
        xml_annotations.write(xml_content)
        xml_annotations.close()
        
        # Image deletion
        os.remove(path)

        new_index += 1

    data.to_csv(conf.PATH_IMAGES + conf.FILE_DATA, index=False)