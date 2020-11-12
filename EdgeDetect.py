
import random, math, os
import numpy as np
from PIL import Image
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dir", type=str, default='images', help='Path to the directory to generate the training dataset from')
args = parser.parse_args()

INPUT_DIR = args.dir
IMAGE_DIR = "targets"
EDGE_DIR = "edges"

if not os.path.exists(INPUT_DIR):
    print("input image directory does not exist!")

if not os.path.exists(IMAGE_DIR):
    os.mkdir(IMAGE_DIR)

if not os.path.exists(EDGE_DIR):
    os.mkdir(EDGE_DIR)

ERODE_KERNEL = np.ones((1,1), np.uint8) 
DILATE_KERNEL = np.ones((2,2), np.uint8) 
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])/2
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])/2

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".png") or filename.endswith(".jpg"): 
        img = cv2.imread(os.path.join(INPUT_DIR,filename))
        img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
        for rotate in range(4):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


            edge = cv2.GaussianBlur(img, (5, 5), 0)

            img_prewittx = cv2.filter2D(edge, -1, kernelx)
            img_prewitty = cv2.filter2D(edge, -1, kernely)
            edge = img_prewittx + img_prewitty
            edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
            ret2,edge = cv2.threshold(edge,0,255,cv2.THRESH_OTSU)

            edge = cv2.erode(edge,ERODE_KERNEL,iterations = 1)
            edge = cv2.dilate(edge, DILATE_KERNEL, iterations=1)

            cv2.imwrite(os.path.join(EDGE_DIR, str(rotate) + filename), edge)
            cv2.imwrite(os.path.join(IMAGE_DIR, str(rotate) + filename), img)
    else:
        continue