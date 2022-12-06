## import the necessary packages
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
# import os
import cv2
import copy
# import pandas as pd
import torch.nn as nn



# The MSER Algorithm
def mser(image, delta=10, min_area=200, max_area=2000):

    # make a copy of the image and convert it to grayscale
    image_ = np.copy(image)
    gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)

    # apply the MSER algorithm to the blurred image
    # Reference: https://docs.opencv.org/4.1.0/d3/d28/classcv_1_1MSER.html

    # mser = cv2.MSER_create(delta=delta, min_area=min_area, max_area=max_area)
    # # mser.setMinArea(1000)
    # # mser.setMaxArea(2000)
    # msers, bboxes = mser.detectRegions(gray)

    # # loop over the regions
    # [[[
    mser = cv2.MSER_create()

    _, bboxes = mser.detectRegions(gray)

    bboxes, _ = cv2.groupRectangles(bboxes.tolist(), groupThreshold=8, eps=0.2)

    bbox_list = []
    for bbox in bboxes:
        x, y, w, h = bbox
        cond1 = 0.3 <= w / h <= 0.8
        cond2 = w > 10 and h > 20
        if cond1 and cond2:
            cv2.rectangle(image_, (x, y), (x + w, y + h), (255,0,0), 3)
            bbox_list.append((x, y, w, h))

    bboxes = np.array(bbox_list)
    # ]]]
    # [[[[[[[[[[

    # color = (255, 0, 0)
    # thickness = 5
    # for bbox in bboxes:
    #     start_point = (bbox[0], bbox[1])
    #     end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    #     cv2.rectangle(image_, start_point, end_point, color, thickness)

    plt.imshow(image_)
    plt.axis('off')
    # save the image
    plt.savefig('mser.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    # ]]]]]]]]]]

    return bboxes

