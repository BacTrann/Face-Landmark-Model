import pandas as pd
import config
import xml.etree.ElementTree as et
import numpy as np
import cv2
# import keras

# Setting print option
import sys
np.set_printoptions(threshold=sys.maxsize)

# Processing data
# Setting up path variables
DATA_PATH = config.DATA_PATH
TEST_DATA_PATH = config.TEST_DATA_PATH
TRAIN_DATA_PATH = config.TRAIN_DATA_PATH


# transform_xml helper function to convert dictionary object to list object
# @param: dict_arr: dictionary array
# @return: list int object
def dict_to_arr(dict_arr):
    value_arr = [list(dict.values()) for dict in dict_arr]
    new_arr = [[int(e[1]), int(e[2])] for e in value_arr]
    return new_arr


# Convert xml data into panda DataFrame
# @param: doc_path : str (file path to xml file)
# @return: panda DataFrame [file, width, height, parts[name, x, y] OR [x,y]]
def transform_xml(doc_path):
    tree = et.parse(doc_path)
    root = tree.getroot()

    data_list = []
    for image in root.findall('images/image'):
        image_info = image.attrib
        box = image.find('box')

        parts = []
        for part in box.findall('part'):
            part_info = part.attrib
            parts.append(part_info)

        # Optional reducing for parts
        parts = dict_to_arr(parts)

        # Add parts to array
        image_info["parts"] = parts
        data_list.append(image_info)

    df = pd.DataFrame(data_list)

    # Convert width and height into integers
    df['file'] = df['file'].map(lambda dir: DATA_PATH + dir)
    df['width'] = df['width'].astype("int")
    df['height'] = df['height'].astype("int")

    return df


# Format data into new panda dataFrame
# @param: data_frame: panda dataFrame to format
# @output: new panda data frame with structure:
#   image(num[][][]) , coord(num[][])
def format_data(data_frame: pd.DataFrame):
    new_frame = pd.DataFrame()

    # Helper function convert picture to numpy array
    # @param: dir: str (directory of picture)
    def file_to_pic_arr(dir):
        image = cv2.imread(dir)
        img_arr = cv2.resize(image, (config.WIDTH, config.HEIGHT))
        return np.array(img_arr, dtype='float32')

    # Helper function return array of rescaled pictures
    # @param:
    #   coord_arr: num[][][] (array of 67 (x, y) coordinates from original picture)
    #   width_arr: num[] (array of original width of pictures)
    #   height_arr: num[] (array of original height of pictures)

    def rescale_coord(coord_arr, width_arr, height_arr):
        new_coords = []
        for i in range(len(coord_arr)):

            def func(coord):
                return [coord[0] * (width_arr[i]/config.WIDTH), coord[1] * (height_arr[i]/config.HEIGHT)]

            for coord in coord_arr:
                new_coords.append(list(map(func, coord)))

        return new_coords

    new_frame['image'] = data_frame['file'].map(file_to_pic_arr)
    new_frame['coord'] = rescale_coord(
        data_frame['parts'], data_frame['width'], data_frame['height'])

    return new_frame


# Save dataFrame to csv file in './processed_data'
# @param:
#   data_frame: pd.DataFrame (data frame wants to save)
#   dir: str (file name)
def save_data(data_frame: pd.DataFrame, dir: str):
    data_frame.to_parquet(f"./processed_data/{dir}", index=False)


# Get train data
def get_train_data():
    train_path = TRAIN_DATA_PATH
    train_pd = transform_xml(train_path)
    return train_pd


# Get test data
def get_test_data():
    test_path = TEST_DATA_PATH
    test_pd = transform_xml(test_path)
    return test_pd




# Testing
def test():
    test = get_test_data()
    save_data(format_data(test), "./processed_data/test_data")
