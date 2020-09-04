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

from cv2 import cv2 as cv
from PIL import Image
from argparse import ArgumentParser

import pandas as pd
import os
import glob
import xmltodict
import json

if __name__ == "__main__":

    # Arguments.
    parser = ArgumentParser()
    
    parser.add_argument("-c", "--color", 
        dest="color",
        help="color",
        type= str,
        required=True)

    args = parser.parse_args()
    
    path = 'images/'
    height = 768
    width = 1024
    color = args.color
    
    if color != 'c' and color != 'g':
        print("No valid color")
        quit()
    
    # Paths.
    resized_path = path + '02_resized/'
    def_path = path + '03_train/'
    
    # Dataframe.
    df_images = pd.read_csv(path + "data.csv")
    
    # Resize ratio.
    x_ratio = width/1920
    y_ratio = height/1080

    for index, row in df_images.copy().iterrows():
        
        image_id = row['name'].split('.')[0]
        
        print("Loading " + image_id)

        # Normal image.
        img = cv.imread(cv.samples.findFile(resized_path + "imgs/" + row["name"]))
        
        if color == 'g':
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        img = cv.resize(img, (width, height))
        cv.imwrite(def_path + "imgs/" + row["name"], img)
        
        # XML read.
        xml = open(resized_path + "coordinates/" + str(image_id) + ".xml", "r")
        xml_content = xml.read()
        boxes = json.loads(json.dumps(xmltodict.parse(xml_content)))['annotation']['object']['bndbox']
        xml.close()
        
        if boxes['xmin'] == None:
            boxes['xmin'] = ''
            boxes['xmax'] = ''
            boxes['ymin'] = ''
            boxes['ymax'] = ''
        else:
            # XML annotations.
            xml_content = xml_content.replace('<xmin>' + boxes['xmin'] + '</xmin>', 
                '<xmin>' + str(int(round(x_ratio * int(boxes['xmin']), 0))) + '</xmin>')
            xml_content = xml_content.replace('<xmax>' + boxes['xmax'] + '</xmax>', 
                '<xmax>' + str(int(round(x_ratio * int(boxes['xmax']), 0))) + '</xmax>')
            xml_content = xml_content.replace('<ymin>' + boxes['ymin'] + '</ymin>', 
                '<ymin>' + str(int(round(y_ratio * int(boxes['ymin']) ,0))) + '</ymin>')
            xml_content = xml_content.replace('<ymax>' + boxes['ymax'] + '</ymax>', 
                '<ymax>' + str(int(round(y_ratio * int(boxes['ymax']) ,0))) + '</ymax>')
        
        # Data replacement.
        xml_content = xml_content.replace('1920', str(width))
        xml_content = xml_content.replace('1080', str(height))
        xml_content = xml_content.replace('png', 'jpg')
        
        # Dataframe coordinates.
        df_images.loc[index, 'coordinates'] = boxes['xmin'] + ';' \
            + boxes['ymin'] + ';' \
            + boxes['xmax'] + ';' \
            + boxes['ymax']

        try:
            coordinates = open(def_path + "coordinates/" + str(image_id) + ".xml", "x")
        except:
            coordinates = open(def_path + "coordinates/" + str(image_id) + ".xml", "w")

        coordinates.write(xml_content)

        # Flipped image.
        flipped = cv.flip(img, 1)
        cv.imwrite(def_path + "imgs/" + str(image_id) + "_B.jpg", flipped, [int(cv.IMWRITE_JPEG_QUALITY), 90])

        df_images.loc[index, 'flipped_name'] = str(image_id) + "_B.jpg"
        
        # XML save.
        try:
            flipped_coordinates = open(def_path + "coordinates/" + str(image_id) + "_B.xml", "x")
        except:
            flipped_coordinates = open(def_path + "coordinates/" + str(image_id)+ "_B.xml", "w")
            
            
        if boxes['xmin'] != '':
            xmin = width - int(round(x_ratio * int(boxes['xmin']),0))
            xmax = width - int(round(x_ratio * int(boxes['xmax']),0))

            if xmin > xmax:
                tmp = xmax
                xmax = xmin
                xmin = tmp
        
            # XML annotations.
            xml_content = xml_content.replace('<xmin>' + str(int(round(x_ratio * int(boxes['xmin']),0)))  + '</xmin>', 
                '<xmin>' + str(xmin) + '</xmin>')
            xml_content = xml_content.replace('<xmax>' + str(int(round(x_ratio * int(boxes['xmax']),0)))  + '</xmax>', 
                '<xmax>' + str(xmax) + '</xmax>')
            xml_content = xml_content.replace(str(image_id) + ".jpg", str(image_id) + "_B.jpg")
            
            # Dataframe coordinates.
            df_images.loc[index, 'flipped_coordinates'] = str(xmin) + ';' \
                + boxes['ymin'] + ';' \
                + str(xmax) + ';' \
                + boxes['ymax']
        else:
            df_images.loc[index, 'flipped_coordinates'] = ';;;'
            
        flipped_coordinates.write(xml_content)
        

    df_images.to_csv(path + "data.csv", index=False)

    df_new = pd.DataFrame(columns=['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class_name'])

    current_row = 0

    # Final dataframe.
    for index, row in df_images.copy().iterrows():
        
        # Normal image.
        df_new.loc[current_row, 'image_name'] = row['name']
        
        # Coordinates.
        coords = row['coordinates'].split(';')
        if coords[0] != '':
            df_new.loc[current_row, 'x_min'] = int(x_ratio * int(coords[0]))
            df_new.loc[current_row, 'y_min'] = int(y_ratio * int(coords[1]))
            df_new.loc[current_row, 'x_max'] = int(x_ratio * int(coords[2]))
            df_new.loc[current_row, 'y_max'] = int(y_ratio * int(coords[3]))
            df_new.loc[current_row, 'class_name'] = 'VMS'
        else:
            df_new.loc[current_row, 'x_min'] = -1
            df_new.loc[current_row, 'y_min'] = -1
            df_new.loc[current_row, 'x_max'] = -1
            df_new.loc[current_row, 'y_max'] = -1
            df_new.loc[current_row, 'class_name'] = '-'
        
        current_row += 1
        
        # Flipped image.
        df_new.loc[current_row, 'image_name'] = row['flipped_name']
        
        # Coordinates.
        coords = row['flipped_coordinates'].split(';')
        if coords[0] != '':
            df_new.loc[current_row, 'x_min'] = coords[0]
            df_new.loc[current_row, 'y_min'] = int(y_ratio * int(coords[1]))
            df_new.loc[current_row, 'x_max'] = coords[2]
            df_new.loc[current_row, 'y_max'] = int(y_ratio * int(coords[3]))
            df_new.loc[current_row, 'class_name'] = 'VMS'
        else:
            df_new.loc[current_row, 'x_min'] = -1
            df_new.loc[current_row, 'y_min'] = -1
            df_new.loc[current_row, 'x_max'] = -1
            df_new.loc[current_row, 'y_max'] = -1
            df_new.loc[current_row, 'class_name'] = '-'
        current_row += 1

    df_new.to_csv('train.csv', index=0)