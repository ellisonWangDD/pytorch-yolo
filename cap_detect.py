#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:36:55 2018

@author: wdd
"""


import numpy as np
from detect import detect
import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import cv2




cfgfile = 'cfg/yolov3.cfg'
weightfile = 'yolov3.weights'
cap = cv2.VideoCapture(0)
count = 0
seq = 'test/test_'

m = Darknet(cfgfile)

#    m.print_network()
m.load_weights(weightfile)
print('Loading weights from %s... Done!' % (weightfile))

num_classes = 80
if num_classes == 20:
    namesfile = 'data/voc.names'
elif num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/names'



print(cap.isOpened())
while (cap.isOpened()):
    ret , frame = cap.read()
    if ret == True:
        file = seq+str(count)+'.jpg'
        cv2.imwrite(file, frame)
        count += 1
        detect(m, cfgfile, weightfile, file)
#        os.remove(file)
