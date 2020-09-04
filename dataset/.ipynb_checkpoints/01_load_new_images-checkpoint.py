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

from PIL import Image
from cv2 import cv2 as cv
from argparse import ArgumentParser

import pandas as pd
import glob
import os

if __name__ == "__main__":

    # Paths.
    path = 'images/'
    PATH_ORIGIN = path + '00_original/*'
    PATH_CLEAN = path + '01_clean/'
    PATH_RESIZED = path + '02_resized/'

    # Dataframe.
    df_images = pd.read_csv(path + "data.csv")
    current_images = df_images["original_name"].tolist()
    next_row = len(df_images.index)

    for file in glob.glob(PATH_ORIGIN):
        if file.split('/')[-1] not in current_images:

            # CSV id,original_name,name,coordinates,flipped_name,flipped_coordinates.
            df_images.loc[next_row, 'id'] = str(next_row)
            df_images.loc[next_row, 'original_name'] = file.split('\\')[-1]
            df_images.loc[next_row, 'name'] = str(next_row).zfill(5) + ".jpg"
            df_images.loc[next_row, 'coordinates'] = '-'
            df_images.loc[next_row, 'flipped_name'] = '-'
            df_images.loc[next_row, 'flipped_coordinates'] = '-'

            # Clean.
            img = Image.open(file)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(PATH_CLEAN + str(next_row).zfill(5) + ".jpg", "JPEG", quality=80, optimize=True, progressive=True)

            # Resized.
            img = cv.imread(cv.samples.findFile(PATH_CLEAN + str(next_row).zfill(5) + ".jpg"))
            img = cv.resize(img, (1920, 1080))
            cv.imwrite(PATH_RESIZED + "imgs/" + str(next_row).zfill(5) + ".jpg", img, [int(cv.IMWRITE_JPEG_QUALITY), 90])

            print('Image ' + str(next_row).zfill(5) + ' added!')
        
        next_row += 1

    df_images.to_csv(path + "data.csv", index=False)