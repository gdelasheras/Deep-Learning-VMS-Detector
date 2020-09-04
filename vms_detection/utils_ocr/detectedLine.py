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

import cv2
import math

class DetectedLine:
    """
        This class represents a line detected in a image.
    """
    def __init__(self, rho, theta):
        """
        Class constructor.
        
        @param rho: polar rho.
        @param theta: polar angle.
        """
        self.rho = rho
        self.theta = theta
        self.a = -math.cos(self.theta)/math.sin(self.theta)
        self.b = self.rho / math.sin(self.theta)
        self.pt1 = (-2000, int(self.a * (-2000) + self.b))
        self.pt2 = (+2000, int(self.a * (+2000) + self.b))
        self.angle = math.degrees(math.atan(self.a))
    
    def draw_line(self, img, color, thickness=3):
        """
        This method draws the line in an image.
        
        @param img: original image.
        @param color: line color.
        @param thickness: line thickness.

        @return: The same image with the line drawn.
        """
        return cv2.line(img, self.pt1, self.pt2, color, thickness)
    
    def y_cut(self, y_line):
        """
        This method calculates the y-coordinate with the horizontal line y = y_line.
        
        @param y_line: horizontal line (y = y_line).

        @return: The x-coordinate of the cut-off point.
        """
        x = (y_line - self.b) / self.a
        return x
    
    def x_cut(self, x_line):
        """
        This method calculates the y-coordinate with the vertical line x = x_line.
        
        @param x_line: vertical line (x = x_line).

        @return: The y-coordinate of the cut-off point.
        """
        y = self.a * x_line + self.b
        return y