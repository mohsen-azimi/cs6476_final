import os

import torch
import torchvision
import torchvision.transforms as transforms
from models import MyModel
from utils import mser

import numpy as np
import matplotlib.pyplot as plt
import cv2


#####################################
# Load Dataset from Torchvision Dataset
# Reference: https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN
# https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN


transform = transforms.Compose([transforms.Resize([32, 32]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])



# get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the network
# load VGG16 model from torchvision
# Reference: https://pytorch.org/vision/stable/models.html
# https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg16
# model_arch = 'vgg16'
model_arch = 'MyModel'


if model_arch == 'MyModel':
    model = MyModel()
    model.load_state_dict(torch.load('my_model_state_dict.pt'))
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

    model.load_state_dict(torch.load('vgg16_scratch_state_dict.pt', device)) # load the best model
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
    image_, roi_list = mser(image, filename)


    print("Total ROIs: ", len(roi_list))

    # loop over the ROIs
    best_roi_list, croped_images = [] , []
    for roi in roi_list:
        print("roi: ", roi)
        criterion = 1

        if criterion:
            print("criterion is True!")
            best_roi_list.append(roi)  # (x, y, w, h)
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

    # get the result from the model and show the result
    model.eval()

    print("Getting the Classification Result...")

    croped_images = croped_images.to(device)

    model.zero_grad()

    outputs = model(croped_images)

    _, preds = torch.max(outputs, 1)

    print("Classification is done!")

    #  probability & the score
    probs = torch.nn.functional.softmax(outputs, dim=1)
    scores = torch.max(probs, dim=1)[0]

    # print the results
    print("labels: ", preds)
    print("scores: ", scores)

    bboxes_, scores_, labels_ = [] , [] , []

    for i, pred in enumerate(preds):
        if pred == 10:  # if the prediction is 10,  skip it (it is background)
            continue
        else:
            bboxes_.append(best_roi_list[i])
            scores_.append(scores[i].item())
            labels_.append(pred.item())

    print( "prediciton_labels: ", labels_)
    print( "prediciton_scores: ", scores_)

    if len(bboxes_) > 0:

        # keep the top bboxes with the highest scores (using non-maximum suppression algorithm)

        keep = torchvision.ops.nms(torch.FloatTensor(bboxes_), torch.FloatTensor(scores_), 0.5)
        keep = keep.numpy().tolist()

        tmp_inx = []
        if len(keep) > 0:
            # 1.  combine the bboxes as one digit
            for i_keep in keep:
                for j_keep in keep:
                    c1 = bboxes_[i_keep][0] < bboxes_[j_keep][0]
                    c2 = bboxes_[i_keep][1] < bboxes_[j_keep][1]
                    c3 = bboxes_[i_keep][2] > bboxes_[j_keep][2]
                    c4 = bboxes_[i_keep][3] > bboxes_[j_keep][3]
                    cccc = c1 and c2 and c3 and c4

                    if cccc:
                        if j_keep not in tmp_inx:  # this avoids the duplication
                            tmp_inx.append(j_keep)

            for index in tmp_inx:  # remove the listed indexes
                keep.remove(index)

            final_bboxes, final_scores, final_labels = [], [], []

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
                _, label, center = cv2.kmeans(np.float32(np.array(final_bboxes)), 2, None,
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10,
                                                cv2.KMEANS_RANDOM_CENTERS)

                h, w, _ = image.shape

                tmp_val1 = (center[0][1] + center[0][3])
                tmp_val2 = (center[1][1] + center[1][3])
                cluster_criterion = abs(tmp_val1-tmp_val2) >= 0.2 * h

                zeros = label.ravel() == 0
                ones = label.ravel() == 1
                cluster_0 = np.mean(final_scores[zeros])
                cluster_1 = np.mean(final_scores[ones])

                if cluster_criterion:

                    if cluster_0 > cluster_1:
                        final_bboxes = final_bboxes[zeros]
                        final_scores = final_scores[zeros]
                        final_labels = final_labels[zeros]
                    elif cluster_0 <= cluster_1:
                        final_bboxes = final_bboxes[ones]
                        final_scores = final_scores[ones]
                        final_labels = final_labels[ones]

            for score in final_scores:
                if score < 0.5:
                    # remove the bbox with the score less than 0.5 (low confidence)
                    final_bboxes = np.delete(final_bboxes, np.where(final_scores == score), axis=0)
                    final_scores = np.delete(final_scores, np.where(final_scores == score), axis=0)
                    final_labels = np.delete(final_labels, np.where(final_scores == score), axis=0)

            # Sort the bboxes based on the x-axis
            centers_x = []
            for bbox in final_bboxes:
                centers_x.append((bbox[0] + bbox[2]) / 2)

            ordered_indx = np.argsort(centers_x)

            digits_list = []
            for indx in ordered_indx:
                digits_list.append(str(final_labels[indx]))

            joined_digits = int("".join(digits_list))


            n_boxes = len(final_bboxes)
            x = np.min([final_bboxes[i][0] for i in range(n_boxes)])
            y = np.min([final_bboxes[i][1] for i in range(n_boxes)])
            w = np.max([final_bboxes[i][2] for i in range(n_boxes)])

            print("joined_digits", str(joined_digits))



            cv2.putText(image, str(joined_digits), (int((x + w) / 2), y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 0, 0), 6)

            plt.imshow(image)
            plt.show()


# ##################################################
# if main, run the code
if __name__ == '__main__':
    print("Running main")

    # create a folder if it does not exist
    if not os.path.exists("graded_images"):
        os.makedirs("graded_images")

    original_image = cv2.imread('test_image.png')
    image = np.copy(original_image)

    # # 1. Original image
    filename = 'original'
    print(f"------------\n filename: {filename} \n")
    image = np.copy(original_image)
    cv2.imwrite(f'{filename}_input.png', image)  # save the image with bounding boxes
    cv2.imwrite('graded_images/0.png', image)  # save the image with bounding boxes

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
    cv2.imwrite('graded_images/4.png', image)  # save the image with bounding boxes
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
    cv2.imwrite('graded_images/5.png', image)  # save the image with bounding boxes
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
    cv2.imwrite('graded_images/1.png', image)  # save the image with bounding boxes

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
    cv2.imwrite('graded_images/2.png', image)  # save the image with bounding boxes

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
    cv2.imwrite(f'graded_images/3.png', image)  # save the image with bounding boxes
    # # #
    # # #
    # # # # load video
    # # # # cap = cv2.VideoCapture('test_video.mp4')
    # # #
    # # #

    # create a folder "graded_images"



