import random
import imutils
import numpy as np
from math import *
from PIL import Image
import face_config

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


class Transform():
    def __init__(self) -> None:
        pass

    # Rotate image by angle
    def rotate(self, image, landmarks, angle):
        # Return a random floating number between +- angle
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))],
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        # Convert the image into a numpy array and rotate by angle
        image = imutils.rotate(np.array(image), angle)

        # Rotating landmarks with picuture
        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    # Resize the image
    # Does not modify landmarks
    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    # Jitter color of image
    def color_jitter(self, image, landmarks):
        # Function randomly jitter color from -param to +param
        color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    # Crop image to only around face
    def crop_face(self, image, landmarks, crops):
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        # Move landmark to new position in cropped picture
        landmarks = torch.tensor(landmarks) - torch.tensor([left, top])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    # Return new image after calling all transformation fucntions above
    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        # Resize image to 224x224
        image, landmarks = self.resize(
            image, landmarks, (face_config.WIDTH, face_config.HEIGHT))
        image, landmarks = self.color_jitter(image, landmarks)
        # Rotate iamge between +- 10 degrees
        image, landmarks = self.rotate(image, landmarks, angle=10)

        # Return transformed image
        image = TF.to_tensor(image)
        # Normalize data to have mean and standard deviation of 0.5
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks
