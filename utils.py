## import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import cv2


# The MSER Algorithm
def mser(image, filename):
    # make a copy of the image and convert it to grayscale
    image_ = np.copy(image)
    gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)

    # Reference: https://docs.opencv.org/4.1.0/d3/d28/classcv_1_1MSER.html

    mser = cv2.MSER_create()

    _, bboxes = mser.detectRegions(gray)

    bboxes, _ = cv2.groupRectangles(bboxes.tolist(), groupThreshold=6, eps=0.5)

    bbox_list = []
    for bbox in bboxes:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x_ = x + w
        y_ = y + h
        if 0.15 <= w / h <= 0.85 and w > 15 and h > 15:
            cv2.rectangle(image_, (x, y), (x_, y_), (255, 0, 0), 3)
            bbox_list.append((x, y, w, h))

    bboxes = np.array(bbox_list)

    plt.imshow(image_)
    plt.axis('off')
    # save the image
    plt.savefig(f'{filename}_smer.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    return image_, bboxes
