
## import the necessary packages
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import copy
import pandas as pd
import PIL



## Define the folders to load the dataset/save the model
DIR_dataset = './dataset'

## get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Line 19: Device: ", device)

## ################ Datast Loading ################
## define the transform
 #reference: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
transform = transforms.Compose([transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    transforms.RandAugment(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),

    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

## get the dataset from torchvision
svhn_train_set = torchvision.datasets.SVHN(root=DIR_dataset, split='train', download=True, transform=transform)
svhn_extra_set = torchvision.datasets.SVHN(root=DIR_dataset, split='extra', download=True, transform=transform)
svhn_test_set = torchvision.datasets.SVHN(root=DIR_dataset, split='test', download=True, transform=transform)

# combine train and extra as train dataset
combined_dataset = torch.utils.data.ConcatDataset([svhn_train_set, svhn_extra_set, svhn_test_set])


## split the train dataset into train and validation dataset
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size

# set seed for reproducibility
torch.manual_seed(0)
train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])

# print the length of train and validation dataset
print("Length of train dataset: ", len(train_dataset))
print("Length of validation dataset: ", len(val_dataset))


## get the dataloader
BATCH_SIZE = 1024
NUM_WORKERS = 2

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


###################### Model Loading ######################
## load VGG16 model from torchvision
PRE_TRAINED = False

if PRE_TRAINED:
    model = torchvision.models.vgg16(pretrained=True)
    file_name = 'vgg16_pretrained.pt'
    # freeze the parameters
    for param in model.parameters():
        param.requires_grad = False
else:
    model = torchvision.models.vgg16(pretrained=False)  # load the model from scratch
    file_name = 'vgg16_scratch.pt'

# print the file name
print("Line 81: File name: ", file_name)


# change the last layer to 10 classes
NUM_CLASSES = 11  # 0-9, 10 for unknown/null
IN_FEATURES = model.classifier[6].in_features  # get the number of features in the last layer
# drop the last layer
features = list(model.classifier.children())[:-1]
# add the new layer
features.extend([torch.nn.Linear(IN_FEATURES, NUM_CLASSES)])

# put all the layers together
model.classifier = torch.nn.Sequential(*features)  # just replaced the last layer to 11 classes
# model.classifier[6] = torch.nn.Sequential(torch.nn.Linear(4096, NUM_CLASSES))

# print the model
print(model)

# move the model to device
model = model.to(device)

# define the loss function
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


# define the number of epochs
NUM_EPOCHS = 15

# define the best accuracy and best model weights to save
best_acc = 0.0

# define the lists to save the loss and accuracy for plotting the graph later
train_loss_list = torch.zeros(NUM_EPOCHS)
train_acc_list = torch.zeros(NUM_EPOCHS)
val_loss_list = torch.zeros(NUM_EPOCHS)
val_acc_list = torch.zeros(NUM_EPOCHS)


## ################ Training ################
for epoch in range(NUM_EPOCHS):
    print("---------------------")

    # # set the model to train mode
    # model.train()

    # define train and val loss and accuracy
    batch_train_loss = 0.0
    batch_train_acc = 0.0
    batch_val_loss = 0.0
    batch_val_acc = 0.0


    # iterate over the data
    for batch, data_sample in enumerate(train_loader, 0):

        # first, get the inputs and labels and send them to device(GPU)
        inputs, labels = data_sample
        inputs = inputs.to(device)
        labels = labels.to(device)

        # require the gradient for the inputs
        inputs.requires_grad = True

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history in train mode
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)

        # backward
        loss.backward()

        # optimize
        optimizer.step()

        # statistics for loss and accuracy

        batch_train_loss += loss.item()
        batch_train_acc += torch.sum(preds == labels.data)


        # print the loss and accuracy to see the progress of the training process
        if batch % 10 == 9:
            print(f"Epoch: {epoch+1},  Batch: {batch+1}, Loss: {loss.item():.4f}, Accuracy: {torch.sum(preds == labels.data).item()/BATCH_SIZE:.4f}")

        # clear the vars from the memory to speed up the training
        del inputs,  outputs, labels, preds, loss
        torch.cuda.empty_cache()

    # calculate the epoch loss and accuracy for train and validation dataset and save them to the lists for plotting the graph later
    epoch_train_loss = batch_train_loss / len(train_dataset)
    epoch_train_acc = batch_train_acc.double() / len(train_dataset)

    # calculate the validation loss and accuracy
    model.eval()
    with torch.no_grad():
        for batch, data_sample in enumerate(val_loader, 0):
            # first, get the inputs and labels
            inputs, labels = data_sample

            # send them to device(GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            optimizer.zero_grad()
            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            # get the results
            batch_val_loss += loss.item()
            batch_val_acc += torch.sum(preds == labels.data)

            # clear the vars from the memory to speed up the training and validation process for the next epoch
            del inputs, outputs, labels,  preds, loss
            torch.cuda.empty_cache()

    epoch_val_loss = batch_val_loss / len(val_dataset)
    epoch_val_acc = batch_val_acc.double() / len(val_dataset)

    # add the loss and accuracy to the list
    train_loss_list[epoch] = epoch_train_loss
    train_acc_list[epoch] = epoch_train_acc
    val_loss_list[epoch] = epoch_val_loss
    val_acc_list[epoch] = epoch_val_acc



    # print the loss and accuracy for each epoch
    print(f"Epoch: {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_acc:.4f}")

    # save the best model weights to the file to use it later for testing
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc

        # save the model
        torch.save(model.state_dict(), f'{file_name}_state_dict.pt')

    # early stopping if the validation loss is not decreasing for 5 epochs
    if epoch > 5:
        if val_loss_list[epoch] > val_loss_list[epoch - 1] > val_loss_list[epoch - 2] > val_loss_list[epoch - 3] > \
                val_loss_list[epoch - 4] > val_loss_list[epoch - 5]:
            print("Early stopping at epoch: ", epoch)
            break


    # save the training history to pandas dataframe
    train_history = pd.DataFrame({'train_loss': train_loss_list, 'train_acc': train_acc_list, 'val_loss': val_loss_list, 'val_acc': val_acc_list})
    train_history.to_csv("train_history.csv", index=False)







