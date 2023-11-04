from face_model import FaceLandModel
from dataset import getTrainData
from utils import print_overwrite
import numpy as np
import face_config
import time

import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    # Setting up model
    model = FaceLandModel()
    model.cuda()

    # Setting loss function as Mean Squared Error
    criterion = nn.MSELoss()
    # Setting optimizer as Adam
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Initialize minimum loss of model as infinity
    loss_min = np.inf
    # Get number of epoch from config
    num_epochs = face_config.EPOCH

    start_time = time.time()

    # Load train, valid data from dataset
    train_loader, valid_loader = getTrainData()

    # Start training
    for epoch in range(1, num_epochs+1):
        loss_train = 0
        loss_valid = 0
        running_loss = 0

        # Set model to training mode
        model.train()

        for step in range(1, len(train_loader)+1):
            # Create data iterable
            images, landmarks = next(iter(train_loader))

            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0), -1).cuda()

            # Calls forward function in model
            predictions = model(images)

            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # Calculate loss of current step
            loss_train_step = criterion(predictions, landmarks)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train/step

            print_overwrite(step, len(train_loader), running_loss, 'train')

        # Set model to non training mode
        model.eval()

        with torch.no_grad():

            for step in range(1, len(valid_loader)+1):

                images, landmarks = next(iter(valid_loader))

                images = images.cuda()
                landmarks = landmarks.view(landmarks.size(0), -1).cuda()

                predictions = model(images)

                # Calculate loss value for the current step using the mean loss function
                loss_valid_step = criterion(predictions, landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        # Print process on console
        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(
            epoch, loss_train, loss_valid))
        print('--------------------------------------------------')

        # Update model if current epoch's model gives better prediction than previous epoch
        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(model.state_dict(), './saved_model/face_landmarks.pth')
            print(
                "\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')

    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time()-start_time))
