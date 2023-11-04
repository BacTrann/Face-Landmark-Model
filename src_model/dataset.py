import face_config
import xml.etree.ElementTree as ET
import numpy as np
import os
import cv2
from transform import Transform


from torch.utils.data import Dataset, DataLoader, random_split


class FaceLandDataSet(Dataset):
    def __init__(self, dataDir, transform=None):
        tree = ET.parse(dataDir)
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = face_config.DATA_PATH

        # Processing data of each image
        # for image in images tag
        for image in root[2]:
            self.image_filenames.append(os.path.join(
                self.root_dir, image.attrib['file']))

            # Values of bounding box for image crop
            self.crops.append(image[0].attrib)

            # Processing 67 landmarks
            landmark = []
            for i in range(68):
                x_coord = int(image[0][i].attrib['x'])
                y_coord = int(image[0][i].attrib['y'])
                landmark.append([x_coord, y_coord])

            # Appending to landmarks
            self.landmarks.append(landmark)

        # Convert landmarks to float32 numpy array
        self.landmarks = np.array(self.landmarks).astype('float32')

        # Check if filenames is equal to landmark array
        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # Use cv2 to read image to gray scale
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        # Apply transfomation function if present
        if self.transform:
            image, landmarks = self.transform(
                image, landmarks, self.crops[index])

        # Zero center input
        landmarks = landmarks - 0.5

        return image, landmarks


def getTrainData():
    dataSet = FaceLandDataSet(face_config.TRAIN_DATA_PATH, Transform())

    # Splitting train and validation set
    # split the dataset into validation and test sets
    len_valid_set = int(0.1*len(dataSet))
    len_train_set = len(dataSet) - len_valid_set

    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))

    train_dataset, valid_dataset,  = random_split(
        dataSet, [len_train_set, len_valid_set])

    # shuffle and batch the datasets
    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8,
                              shuffle=True, num_workers=4)

    return train_loader, valid_loader
