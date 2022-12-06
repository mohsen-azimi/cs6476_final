"""
[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
good refs: https://github.com/GrafDuckula/CS6476-Computer-Vision/blob/main/Final_project/Detection_on_video.ipynb
]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
"""
## import the necessary packages
import torch
import torchvision
import torchvision.transforms as transforms
# import torchvision.functional as F
from models import MyModel
from utils import mser

import numpy as np
import matplotlib.pyplot as plt
# import os
import cv2
import copy

# import pandas as pd


#####################################
# Load Dataset from Torchvision Dataset
# Reference: https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN
# https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN


# define the transform
# transform = transforms.Compose([transforms.CenterCrop((32, 32)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                 ])

transform = transforms.Compose([transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])


# [[[
# transform = transforms.Compose([
#                 transforms.Resize([48, 48]),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])]) # orignally for ImageNet
# ]]]

# get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the network
# load VGG16 model from torchvision
# Reference: https://pytorch.org/vision/stable/models.html
# https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg16
model_arch = 'vgg16'
# model_arch = 'MyModel'


if model_arch == 'MyModel':
    model = MyModel()
    model.load_state_dict(torch.load('saved_models/my_model.pt'))
    model.to(device)
    model.eval()
    print(f"{model_arch} is loaded successfully!")

else:
    model = torchvision.models.vgg16(pretrained=False)
    #
    IN_FEATURES = model.classifier[6].in_features  # get the number of features in the last layer
    features = list(model.classifier.children())[:-1]
    features.extend([torch.nn.Linear(IN_FEATURES, 11)])
    model.classifier = torch.nn.Sequential(*features)  # just replaced the last layer to 11 classes

    model.load_state_dict(torch.load('saved_models/vgg16_scratch_state_dict.pt', device)) # load the best model
    # model = torch.load('saved_models/best_model_vgg16_from_scratch.pt')  # entire model
    model.to(device)
    model.eval()
    print(f"{model_arch} is loaded successfully!")

# define the loss function
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def run_detection_and_classification(image, filename):
    print("run_detection_and_classification() is called!")
    # get the roi using MSER
    roi_list = mser(image, filename)
    print("Total ROIs: ", len(roi_list))

    # loop over the regions
    best_roi_list = []
    croped_images = []
    for roi in roi_list:
        print("roi: ", roi)
        # x1 = roi[0] - roi[2]
        # y1 = roi[1] - roi[3]
        # x2 = roi[0] + roi[2] + roi[2]
        # y2 = roi[1] + roi[3] + roi[3]
        #
        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #
        # x1 = roi[0] if x1 < 0 else x1
        # y1 = roi[1] if y1 < 0 else y1
        # x2 = roi[0] + roi[2] if x2 > image.shape[1] else x2
        # y2 = roi[1] + roi[3] if y2 > image.shape[0] else y2

        criterion = 1 #= 10 <= roi[2] <= 1000 and 10 <= roi[3] <= 1000 and roi[3] / roi[2] <= 3. and roi[2] / roi[3] <= 3.

        if criterion:
            print("criterion is True!")
            # best_roi_list.append((x1, y1, x2, y2))
            best_roi_list.append(roi)  # (x, y, w, h)
            # image_cut = image[y1:y2, x1:x2, :]
            # image_cut = torchvision.transforms.functional.to_pil_image(image_cut)
            # image_cut = transform(image_cut)

            print("shape of image to cut: ", image.shape)
            cropped_image = image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
            # show the image
            imgg = transform(torchvision.transforms.functional.to_pil_image(cropped_image))
            imgg = torch.transpose(imgg, 0, 2)
            imgg = torch.transpose(imgg, 0, 1)
            print("shape of cropped image: ", imgg.shape)
            plt.imshow(imgg)
            plt.axis('off')
            # save the image
            plt.savefig(f'{filename}_smer_cropped_{len(best_roi_list)}.png', bbox_inches='tight', pad_inches=0)
            plt.show()

            croped_images.append(transform(torchvision.transforms.functional.to_pil_image(cropped_image)))

    # convert the list to tensor
    croped_images = torch.stack(croped_images)

    # get the prediction
    # model.train(False)
    model.eval()

    # print("Getting the prediction...")

    croped_images = croped_images.to(device)

    model.zero_grad()

    outputs = model(croped_images)

    _, preds = torch.max(outputs, 1)

    print("Prediction is done!")

    # get the probability & the score
    probs = torch.nn.functional.softmax(outputs, dim=1)
    scores = torch.max(probs, dim=1)[0]

    # print the results
    print("labels: ", preds)
    print("scores: ", scores)

    bboxes_ = []
    scores_ = []
    labels_ = []

    for i, pred in enumerate(preds):
        if pred == 10:  # if the prediction is 10,  skip it (it is background)
            continue
        else:
            bboxes_.append(best_roi_list[i])
            scores_.append(scores[i].item())
            labels_.append(pred.item())

    print( "prediciton_labels: ", labels_)
    print( "prediciton_scores: ", scores_)


    # [[[ https://github.com/GrafDuckula/CS6476-Computer-Vision/blob/5357d75d7adef2622231601489029c07b4c63149/Final_project/run.py#L446

    if len(bboxes_) > 0:

        # keep the top N d
        # etections
        keep = torchvision.ops.nms(torch.FloatTensor(bboxes_), torch.FloatTensor(scores_), 0.5)
        keep = keep.numpy().tolist()

        tmp_inx = []
        if len(keep) > 0:
            # 1.  combine the bboxes as one digit
            for i_keep in keep:
                for j_keep in keep:
                    criterion1 = bboxes_[i_keep][0] < bboxes_[j_keep][0]
                    criterion2 = bboxes_[i_keep][1] < bboxes_[j_keep][1]
                    criterion3 = bboxes_[i_keep][2] > bboxes_[j_keep][2]
                    criterion4 = bboxes_[i_keep][3] > bboxes_[j_keep][3]
                    criteria = criterion1 and criterion2 and criterion3 and criterion4

                    if criteria:
                        if j_keep not in tmp_inx:
                            tmp_inx.append(j_keep)

            for index in tmp_inx:
                keep.remove(index)

            final_bboxes = []
            final_scores = []
            final_labels = []
            for ind in keep:
                final_bboxes.append(bboxes_[ind])
                final_scores.append(scores_[ind])
                final_labels.append(labels_[ind])

            # convert the list to numpy array
            final_bboxes = np.array(final_bboxes)
            final_scores = np.array(final_scores)
            final_labels = np.array(final_labels)


            if len(keep) > 1:

                print("len of keep: ", len(keep))

                # clustering the bboxes
                # kmeans_random_centers = cv2.KMEANS_RANDOM_CENTERS
                ret, label, center = cv2.kmeans(np.float32(np.array(final_bboxes)), 2, None,
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10,
                                                cv2.KMEANS_RANDOM_CENTERS)

                # print(f"label: {label}")
                # print(f"center: {center}")
                h, w, _ = image.shape

                tmp_val = center[0][1] + center[0][3] - center[1][1] - center[1][3]
                cluster_criterion = abs(tmp_val) >= 0.2 * h

                cluster_0 = np.mean(final_scores[label.ravel() == 0])
                cluster_1 = np.mean(final_scores[label.ravel() == 1])

                if cluster_criterion:

                    if cluster_0 > cluster_1:
                        final_bboxes = final_bboxes[label.ravel() == 0]
                        final_scores = final_scores[label.ravel() == 0]
                        final_labels = final_labels[label.ravel() == 0]
                    else:
                        final_bboxes = final_bboxes[label.ravel() == 1]
                        final_scores = final_scores[label.ravel() == 1]
                        final_labels = final_labels[label.ravel() == 1]

            for score in final_scores:
                if score < 0.5:
                    # remove the bbox
                    final_bboxes = np.delete(final_bboxes, np.where(final_scores == score), axis=0)
                    final_scores = np.delete(final_scores, np.where(final_scores == score), axis=0)
                    final_labels = np.delete(final_labels, np.where(final_scores == score), axis=0)

            # [[[[
            x_centers = [(final_bboxes[idx][0] + final_bboxes[idx][2]) / 2 for idx in range(len(final_bboxes))]
            digit_order = np.argsort(x_centers)
            digits = [str(final_labels[i]) for i in digit_order]
            final_number = int("".join(digits))

            final_x1 = np.min([final_bboxes[idx][0] for idx in range(len(final_bboxes))])
            final_y1 = np.min([final_bboxes[idx][1] for idx in range(len(final_bboxes))])
            final_x2 = np.max([final_bboxes[idx][2] for idx in range(len(final_bboxes))])
            final_y2 = np.max([final_bboxes[idx][3] for idx in range(len(final_bboxes))])

            print("final_numbers", str(final_number))

            number, (x1, y1, x2, y2) = final_number, (final_x1, final_y1, final_x2, final_y2)

            # ]]]]


            # 2.  create the output image
            # [[[[[

            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 10)
            cv2.putText(image, str(final_number), (int((x1 + x2) / 2), y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 0, 0), 5)

            plt.imshow(image)
            plt.show()

    # ]]]]]]]]]


# ##################################################
# if main, run the code
if __name__ == '__main__':
    print("Running main")
    original_image = cv2.imread('test_image.png')
    image = np.copy(original_image)

    # # 1. Original image
    filename = 'original'
    print(f"------------\n filename: {filename} \n")
    image = np.copy(original_image)
    cv2.imwrite(f'{filename}_input.png', image)  # save the image with bounding boxes

    run_detection_and_classification(image, filename)  # detect and classify the numbers in the image
    cv2.imwrite(f'{filename}_output.png', image)  # save the image with bounding boxes
    # #
    # #
    # # # 2. Change brightness
    del image
    filename = 'brightness'
    image = np.copy(original_image)
    image = (image * 0.7).clip(0, 255).astype(np.uint8)
    cv2.imwrite(f'{filename}_input.png', image)  # save the image with bounding boxes

    run_detection_and_classification(image, filename)  # detect and classify the numbers in the image
    cv2.imwrite(f'{filename}_output.png', image)  # save the image with bounding boxes
    # #
    # #
    # # # 3. add noise
    del image
    filename = 'noise_5'
    image = np.copy(original_image) + np.random.normal(0, 5, original_image.shape).astype(np.uint8) # add noise
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(f'{filename}_input.png', image)  # save the image with bounding boxes

    run_detection_and_classification(image, filename)  # detect and classify the numbers in the image
    cv2.imwrite(f'{filename}_output.png', image)  # save the image with bounding boxes
    # #
    # #
    del image
    filename = 'noise_10'
    image = np.copy(original_image) + np.random.normal(0, 10, original_image.shape).astype(np.uint8) # add noise
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(f'{filename}_input.png', image)  # save the image with bounding boxes

    run_detection_and_classification(image, filename)  # detect and classify the numbers in the image
    cv2.imwrite(f'{filename}_output.png', image)  # save the image with bounding boxes
    # #

    # # # 4. scale  image
    del image
    filename = 'scale'
    image = np.copy(original_image)
    image = cv2.resize(image, (0, 0), fx=1.4, fy=1.4)
    # center-crop the image to the original size
    h_o, w_o, _ = original_image.shape
    h, w, _ = image.shape
    image = image[int((h - h_o) / 2):int((h - h_o) / 2) + h_o, int((w - w_o) / 2):int((w - w_o) / 2) + w_o]

    cv2.imwrite(f'{filename}_input.png', image)  # save the image with bounding boxes

    run_detection_and_classification(image, filename)  # detect and classify the numbers in the image
    cv2.imwrite(f'{filename}_output.png', image)  # save the image with bounding boxes
    # # #
    # # #
    # # #
    # # 5. rotate image 20 degrees
    del image
    filename = 'rotate_20'
    image = np.copy(original_image)
    from scipy import ndimage
    image = ndimage.rotate(image, 20)
    cv2.imwrite(f'{filename}_input.png', image)  # save the image with bounding boxes

    run_detection_and_classification(image, filename)  # detect and classify the numbers in the image
    cv2.imwrite(f'{filename}_output.png', image)  # save the image with bounding boxes
    # # # #

    # # 5. rotate image 20 degrees
    # # # #
    # # # 6. move location
    del image
    filename = 'move'
    image = np.copy(original_image)
    image = cv2.warpAffine(image, np.float32([[1, 0, 200], [0, 1, 200]]), (image.shape[1], image.shape[0]))
    cv2.imwrite(f'{filename}_input.png', image)  # save the image with bounding boxes
    run_detection_and_classification(image, filename)  # detect and classify the numbers in the image
    cv2.imwrite(f'{filename}_output.png', image)  # save the image with bounding boxes
    # # #
    # # #
    # # # # load video
    # # # # cap = cv2.VideoCapture('test_video.mp4')
    # # #
    # # #
