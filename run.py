
"""
good refs:
https://github.com/GrafDuckula/CS6476-Computer-Vision/blob/main/Final_project/Detection_on_video.ipynb
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2




# Load Dataset from Torchvision Dataset
# Reference: https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN
# https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN


# define the transform
transform = transforms.Compose([transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# get the dataset from torchvision
train_datset = torchvision.datasets.SVHN(root='./dataset', split='train', download=True, transform=transform)
test_dataset = torchvision.datasets.SVHN(root='./dataset', split='test', download=True, transform=transform)
extra_dataset = torchvision.datasets.SVHN(root='./dataset', split='extra', download=True, transform=transform)

train_dataset = torch.utils.data.ConcatDataset([train_datset, extra_dataset]) # combine train and extra as train dataset


# get the dataloader
train_loader = torch.utils.data.DataLoader(train_datset, batch_size=4, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# get the classes
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# trainloader to device
train_loader = train_loader.to(device)
test_loader = test_loader.to(device)

# define the network
# load VGG16 model from torchvision
# Reference: https://pytorch.org/vision/stable/models.html
# https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg16

model = torchvision.models.vgg16(pretrained=True)

# define the loss function
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# train the model
num_epochs = 10
for epoch in range(num_epochs):
    # run_loss = 0.
    for batch, data_sample in enumerate(train_loader, 0):

        # first, get the inputs and labels and send them to device(GPU)
        inputs, labels = data_sample
        inputs = inputs.to(device)
        labels = labels.to(device)

        # second, zero the parameter gradients (need to do this every time, because Pytorch says!)
        optimizer.zero_grad()

        # third, forward -> backward -> optimize
        outputs = model(inputs)

        # get the loss and perform backpropagation as well as optimization
        loss = criterion(outputs, labels)
        loss.backward()

        # finally, update the parameters of the model (i.e., the weights by performing a gradient descent step)
        optimizer.step()

        # show the training history (loss)
        print(f"Epoch: {epoch+1}, Batch: {batch+1}, Loss: {loss.item():.4f}")


Now, Let's test the model, but first, let's save the model
Reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html

# save the model
torch.save(model.state_dict(), './model.pt')

# load the model from model.pt
model = torch.load("model.pt")

# test the model
# set the model to evaluation mode
model.eval()

# calculate the accuracy
predictions = []

with torch.no_grad():
    for data_sample in test_loader:
        inputs, labels = data_sample
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, y_preds = torch.max(outputs, 1)

        predictions.append(y_preds == labels)

    accuracy = torch.cat(predictions).float().mean()

    print(f"Test Accuracy: {accuracy.item():.4f}")



# [[[[[[[[[[[[[[[[[
# Now, let's test the model with a single image

# load the image
img = cv2.imread('./test.png')

# convert the image to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# show the image
plt.imshow(img)
plt.show()

# convert the image to tensor
img = torch.from_numpy(img).float()

# normalize the image
img = img / 255.0

# resize the image
img = img.resize_(1, 3, 32, 32)

# send the image to device
img = img.to(device)

# set the model to evaluation mode
model.eval()

# get the prediction
with torch.no_grad():
    outputs = model(img)

    _, y_preds = torch.max(outputs, 1)

    print(f"Prediction: {y_preds.item()}")


#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]




# # test the model
# data_iter = iter(test_loader)
# inputs, labels = data_iter.next()
#
#
# outputs = model(inputs)
# _, y_preds = torch.max(outputs, 1)
#
# for data_sample in test_loader:
#
#     inputs, labels = data_sample
#
#     # get the outputs from the trained model
#     outputs = model(inputs)
#
#     # get the prediction from the outputs (i.e., the class with the highest probability)
#     _, y_preds = torch.max(outputs, 1)
#
#
#















# ##################################################
# if main, run the code
if __name__ == '__main__':
    print("Running main")

