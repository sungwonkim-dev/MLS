import tensorflow as tf
import cv2 as cv2
import numpy as np


def convert_img_to_projection(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    img = 255 - img
    return height, width, np.sum(img, 1)


if __name__ == '__main__':
    path = "C:\\dev\\Python\\Project\\MLS\\res\\image\\example\\CostFuntion.PNG"
    height, width, projection = convert_img_to_projection(path)
