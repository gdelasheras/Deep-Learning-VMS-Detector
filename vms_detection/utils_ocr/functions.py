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

from utils_ocr.detectedLine import DetectedLine
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def detect_lines(img, preprocessing, threshold):
    """
    This function detects lines in an image.
    
    @param img: Original image.
    @param preprocessing: Indicates whether the image needs to be pre-processed before the canny algorithm.
    @param threshold: Minimum threshold to detect a line in the hough transform.
    
    @return:
        - lines: List of detected lines.
        - img_canny: Canny algorithm output image.
    """
    
    lines = []
    
    # Line detection using the canny algorithm.
    hough_lines, img_canny = get_HoughLines(img, preprocessing, threshold)
    
    if hough_lines is not None:
        for i in range(0, len(hough_lines)):
            if hough_lines[i][0][1] != 0:
                # Appending detected line.
                lines.append(DetectedLine(hough_lines[i][0][0], hough_lines[i][0][1]))
    
    return lines, img_canny

def auto_canny(img, sigma=0.33):
    """
    This function applies the canny algorithm to a given image.
    source: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    
    @param img: Original image.
    @param sigma: Percentage change in threshold.
    
    @return: Canny algorithm output image.
    """
    
    # Compute the median of the single channel pixel intensities.
    median = np.median(img)
    
    # Thresholds.
    lower_threshold = int(max(0, (1.0 - sigma) * median))
    upper_threshold = int(min(255, (1.0 + sigma) * median))
    
    return cv2.Canny(img, lower_threshold, upper_threshold)

def get_HoughLines(img, preprocessing, threshold):
    """
    This function applies the Hough transform to find the parameters of the lines within an image.
    
    @param img: Original image.
    @param preprocessing: Indicates whether the image needs to be pre-processed before the canny algorithm.
    @param threshold: Minimum threshold to detect a line in the hough transform.
    
    @return:
        - hough_lines: list of detected lines expressed in polar coordinates.
        - img_canny: Canny algorithm output image.
    """
    
    if preprocessing:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Canny algorithm
    img_canny = auto_canny(img, sigma=0.33)
    
    # Hough transform.
    hough_lines = cv2.HoughLines(img_canny, 1, np.pi / 180, threshold, None, 0, 0)
    
    return hough_lines, img_canny

def fix_rotation(img, max_angle, min_angle):
    """
    This function straightens a given image.
    
    @param img: Original image.
    @param max_angle: Minimum angle of the lines detected in the hough transform.
    @param min_angle: Maximun angle of the lines detected in the hough transform.
    
    @return:
        - img_rotated: Straightened image.
        - img_lines: Original image with the lines detected in the hough transform drawn.
    """    
    
    degrees = []
    img_lines, img_rotated = img.copy(), img.copy()
    
    lines, img_canny = detect_lines(img, preprocessing=True, threshold=int(img.shape[1]/3))
    
    y_threshold = img.shape[0] / 6
    
    for line in lines:
        if np.abs(line.angle) <= max_angle and np.abs(line.angle) >= min_angle:

            if line.x_cut(0) >= 0 and line.x_cut(0) <= y_threshold:
                line.draw_line(img_lines, (0, 255, 255), 2)
                degrees.append(line.angle)
                
            elif line.x_cut(img.shape[1]) <= img.shape[0] and line.x_cut(img.shape[1]) >= img.shape[0] - y_threshold:
                line.draw_line(img_lines, (0, 255, 255), 2)
                degrees.append(line.angle)
    
    if len(degrees) > 0:
        img_rotated = rotate(img_rotated, round(np.mean(degrees)))
        print("Mean rotation angle: " + str(round(np.mean(degrees))), end="\n\n")
    
    return img_rotated, img_lines

def rotate(img, angle):
    """
    This function rotates a given image.
    
    @param img: Original image.
    @param angle: Rotation angle.
    
    @return: Rotated image.
    """
    
    # Height and width.
    (height, width) = img.shape[:2]
    
    # Rotation matrix.
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    
    # Rotated image.
    img_rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    return img_rotated

def crop_image(img, min_angle, max_angle, preprocessing, threshold_dots, threshold_axis, axis):
    """
    This function crops a given image according to the lines detected in the hough transform.
    
    @param img: Original image.
    @param min_angle: Minumun angle of the lines detected in the hough transform.
    @param max_angle: Maximun angle of the lines detected in the hough transform.
    @param preprocessing: Indicates whether the image needs to be pre-processed before the canny algorithm.
    @param threshold_dots: Minimum threshold to detect a line in the hough algorithm.
    @param threshold_axis: Indicates the range of values on the axis where the lines detected in the hough transform must be found. 
    @param axis: Crop axis.
    
    @return:
        - img_cropped: Cropped image.
        - img_lines: Original image with the detected lines drawn and the cropping lines.
    """
    
    if axis != "v" and axis != "h":
        raise Exception("Unknown axis.")
    
    img_lines = img.copy()
    
    # Lines detection.
    lines, img_canny = detect_lines(img, preprocessing, threshold_dots)
    
    # Horizontal axis.
    if axis == "h":
        img_cropped, img_lines = __horizontal_crop(img_lines, img.copy(), lines, min_angle, max_angle, threshold_axis)
            
    # Vertical axis.
    elif axis == "v":
        img_cropped, img_lines = __vertical_crop(img_lines, img.copy(), lines, min_angle, max_angle, threshold_axis)
                
    return img_cropped, img_lines


def __vertical_crop(img_lines, img_cropped, lines, min_angle, max_angle, x_threshold):
    """
    This function laterally crops a given image according to the vertical lines detected in the hough transformation.
    
    @param img_lines: Original image with the detected lines drawn.
    @param img_cropped: Copy of the original image.
    @param lines: Lines detected in the hough transform.
    @param min_angle: Minumun angle of the lines detected in the hough transform.
    @param max_angle: Maximun angle of the lines detected in the hough transform.
    @param x_threshold: Indicates the range of values on the x-axis where the lines detected in the hough transform must be found. 
    
    @return:
        - img_cropped: Cropped image.
        - img_lines: Original image with the detected lines drawn and the cropping lines.
    """

    left_dots  = [0]
    right_dots = [img_lines.shape[1]]
    
    for line in lines:
        if np.abs(line.angle) <= max_angle and np.abs(line.angle) >= min_angle:
            
            # Left crop.
            if line.y_cut(0) >= 0 and line.y_cut(0) <= x_threshold:
                left_dots.append(int(line.y_cut(0)))
                line.draw_line(img_lines, (0, 255, 255), 2)
            
            # Right lines.
            if line.y_cut(img_lines.shape[0]) <= img_lines.shape[1] \
                and line.y_cut(img_lines.shape[0]) >= img_lines.shape[1] - x_threshold:
                right_dots.append(int(line.y_cut(img_lines.shape[0])))
                line.draw_line(img_lines, (0, 255, 255), 2)
                
    # Right crop.
    right_crop = min(right_dots)
    right_crop = int(right_crop - right_crop * 0.03)
    lines_img = cv2.line(img_lines, (right_crop, 0), (right_crop, img_lines.shape[0]), (255, 0, 255), 3)
    
    # Left crop.
    left_crop = max(left_dots)
    left_crop = int(left_crop + right_crop * 0.03)
    lines_img = cv2.line(img_lines, (left_crop, 0), (left_crop, img_lines.shape[0]), (255, 0, 255), 3)
        
    # [top_edge : bottom_edge, left_edge : right_edge]
    img_cropped = img_cropped[0 : img_cropped.shape[0], left_crop : right_crop]
    
    return img_cropped, img_lines
        
def __horizontal_crop(img_lines, img_cropped, lines, min_angle, max_angle, y_threshold):
    """
    This function crops the top and bottom of a given image according to the horizontal lines detected in the hough transformation.
    
    @param img_lines: Original image with the detected lines drawn.
    @param img_cropped: Copy of the original image.
    @param lines: Lines detected in the hough transform.
    @param min_angle: Minumun angle of the lines detected in the hough transform.
    @param max_angle: Maximun angle of the lines detected in the hough transform.
    @param y_threshold: Indicates the range of values on the y-axis where the lines detected in the hough transform must be found. 
    
    @return:
        - img_cropped: Cropped image.
        - img_lines: Original image with the detected lines drawn and the cropping lines.
    """
    upper_dots = [0]
    lower_dots = [img_lines.shape[0]]
    
    for line in lines:
        if np.abs(line.angle) <= max_angle and np.abs(line.angle) >= min_angle:
            
            # Upper lines.
            if line.x_cut(0) >= 0 and line.x_cut(0) <= y_threshold:
                upper_dots.append(int(line.x_cut(0)))
                line.draw_line(img_lines, (0, 255, 255), 2)
            
            # Lower lines.
            if line.x_cut(img_lines.shape[1]) <= img_lines.shape[0] \
                and line.x_cut(img_lines.shape[1]) >= img_lines.shape[0] - y_threshold:
                lower_dots.append(int(line.x_cut(img_lines.shape[1])))
                line.draw_line(img_lines, (0, 255, 255), 2)
                
    # Lower crop.
    lower_crop = min(lower_dots)
    lower_crop = int(lower_crop - lower_crop * 0.05)
    img_lines = cv2.line(img_lines, (0, lower_crop), (img_lines.shape[1], lower_crop), (255, 0, 255), 3)
                    
    # Upper crop.
    upper_crop = max(upper_dots)
    upper_crop = int(upper_crop + lower_crop * 0.03)
    img_lines = cv2.line(img_lines, (0, upper_crop), (img_lines.shape[1], upper_crop), (255, 0, 255), 3)

    # [top_edge : bottom_edge, left_edge : right_edge]
    img_cropped = img_cropped[upper_crop : lower_crop, 0 : img_cropped.shape[1]]
    
    return img_cropped, img_lines

def prepare_for_ocr(img):
    """
    This function prepares the image for OCR
    
    @param img: Original image.
    
    @return:
        - img: Image ready for OCR.
        - otsu: Image processed by Otsu threshold.
        - close: Image processed by the morphological operation of closing.
    """
    
    # Otsu binarization method.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    otsu = cv2.bitwise_not(img)
    
    # Closing morphological operation.
    img = cv2.GaussianBlur(otsu, (3, 3), 0)
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # Histogram equalization.
    img = cv2.equalizeHist(close)
    
    return img, otsu, close

def show(img, color=True, size=(10, 5)):
    """
    This functions displays a given image.
    
    @param img: Original image.
    @param color: Indicates whether the image is displayed in color.
    @param size: Indicates the image size.
    """
    plt.figure(figsize=size)
    plt.axis("off")
    if color is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.show()
    
def draw_detected_text(img, data):
    """
    
    
    @param img: Original image.
    @param data: 
    """
    img = img.copy()
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if int(data['conf'][i]) >= 0:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
    return img

def plot_results(images):
    """
    
    
    @param images: 
    """
    
    fig, axs = plt.subplots(2, 3, figsize=(20, 5))
    
    axs[0][0].imshow(cv2.cvtColor(images[0][0], cv2.COLOR_BGR2RGB))
    axs[0][0].set_title(images[0][1])
    axs[0][0].axis('off')
    axs[0][1].imshow(cv2.cvtColor(images[1][0], cv2.COLOR_BGR2RGB))
    axs[0][1].set_title(images[1][1])
    axs[0][1].axis('off')
    axs[0][2].imshow(cv2.cvtColor(images[2][0], cv2.COLOR_BGR2RGB))
    axs[0][2].set_title(images[2][1])
    axs[0][2].axis('off')
    axs[1][0].imshow(cv2.cvtColor(images[3][0], cv2.COLOR_BGR2RGB))
    axs[1][0].set_title(images[3][1])
    axs[1][0].axis('off')
    axs[1][1].imshow(cv2.cvtColor(images[4][0], cv2.COLOR_BGR2RGB))
    axs[1][1].set_title(images[4][1])
    axs[1][1].axis('off')
    axs[1][2].imshow(cv2.cvtColor(images[5][0], cv2.COLOR_BGR2RGB))
    axs[1][2].set_title(images[5][1])
    axs[1][2].axis('off')
    
    plt.show()