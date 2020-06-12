import keras
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from os import listdir
from PIL import Image as PImage
from dataset_labelmatcher import get_similar_column
from grammartree import get_value_instruction
import cv2


def initial_preprocesser(data, instruction, preprocess):
    # Scans for object columns just in case we have a datetime column that
    # isn't detected
    object_columns = [
        col for col,
        col_type in data.dtypes.iteritems() if col_type == 'object']

    # Handles dates without timestamps
    for col in object_columns:
        try:
            data[col] = pd.to_datetime(data[col], infer_datetime_format=True)
        except ValueError:
            pass

    # get target column
    target = get_similar_column(
        get_value_instruction(instruction), data)
    y = data[target]

    # remove rows where target is NaN
    data = data[y.notna()]
    y = y[y.notna()]

    del data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=49)

    data = {
        'train': pd.concat([X_train, y_train], axis=1),
        'test': pd.concat([X_test, y_test], axis=1)
    }

    # preprocess the dataset
    full_pipeline = None
    if preprocess:
        data, full_pipeline = structured_preprocesser(data)
    else:
        data.fillna(0, inplace=True)

    return data, y, target, full_pipeline


# Preprocesses the data appropriately for single reg data
def structured_preprocesser(data):

    # Preprocessing for datetime columns
    process_dates(data)

    # identifies the categorical and numerical columns
    categorical_columns = data['train'].select_dtypes(
        exclude=["number"]).columns
    numeric_columns = data['train'].columns[data['train'].dtypes.apply(
        lambda c: np.issubdtype(c, np.number))]

    # pipeline for numeric columns
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

    # pipeline for categorical columns
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="constant", fill_value="")),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    # combine the two pipelines
    if len(numeric_columns) != 0 and len(categorical_columns) != 0:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, numeric_columns),
            ("cat", cat_pipeline, categorical_columns),
        ], remainder='passthrough')
    elif len(numeric_columns) == 0:
        full_pipeline = ColumnTransformer([
            ("cat", cat_pipeline, categorical_columns),
        ])
    else:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, numeric_columns),
        ])

    train = full_pipeline.fit_transform(data['train'])
    train_encoded_cols = full_pipeline.named_transformers_[
        'cat']['one_hot_encoder'].get_feature_names()
    train_cols = [*list(numeric_columns), *train_encoded_cols]

    test = full_pipeline.transform(data['test'])
    test_encoded_cols = full_pipeline.named_transformers_[
        'cat']['one_hot_encoder'].get_feature_names()
    test_cols = [*list(numeric_columns), *test_encoded_cols]

    # Ternary clause because when running housing.csv,
    # the product of preprocessing is np array, but not when using landslide
    # data... not sure why
    data['train'] = pd.DataFrame(
        (train.toarray() if not isinstance(
            train,
            np.ndarray) else train),
        columns=train_cols)
    data['test'] = pd.DataFrame(
        (test.toarray() if not isinstance(
            train,
            np.ndarray) else test),
        columns=test_cols)

    return data, full_pipeline


# Preprocesses images from images to median of heighs/widths
def image_preprocess(data_path, new_folder=True):
    training_path = data_path + "/training_set"
    testing_path = data_path + "/testing_set"

    heights = []
    widths = []
    classification = 0

    training_dict = {}
    for class_folder in listdir(training_path):
        if not os.path.isdir(training_path + "/" + class_folder):
            continue
        training_dict[class_folder] = {}
        classification += 1
        for image in listdir(training_path + "/" + class_folder):
            try:
                img = cv2.imread(training_path + "/" + class_folder + "/" + image)
                heights.append(img.shape[0])
                widths.append(img.shape[1])
                training_dict[class_folder][image] = img
            except BaseException:
                continue

    testing_dict = {}
    for image in listdir(testing_path):
        try:
            img = cv2.imread(testing_path + "/" + image)
            heights.append(img.shape[0])
            widths.append(img.shape[1])
            testing_dict[image] = img
        except BaseException:
            continue

    heights.sort()
    widths.sort()
    height = heights[int(len(heights)/2)]
    width = widths[int(len(widths)/2)]

    # resize images
    for class_folder, images in training_dict.items():
        for image_name, image in images.items():
            training_dict[class_folder][image_name] = process_color_channel(image, height, width)

    for image_name, image in testing_dict.items():
        testing_dict[image_name] = process_color_channel(image, height, width)

    # create new folder containing resized images
    if new_folder:
        # check if proc_training_set folder exists
        if os.path.isdir(data_path + "/proc_training_set"):
            shutil.rmtree(data_path + "/proc_training_set")
        os.mkdir(data_path + "/proc_training_set")
        for class_folder, images in training_dict.items():
            add_resized_images(data_path + "/proc_training_set", class_folder, images)
        # check if proc_testing_set folder exists
        if os.path.isdir(data_path + "/proc_testing_set"):
            shutil.rmtree(data_path + "/proc_testing_set")
        add_resized_images(data_path, "testing_set", testing_dict)
    # replace images with newly resized images
    else:
        for class_folder, images in training_dict.items():
            replace_images(training_path + "/" + class_folder, images)
        replace_images(testing_path + "/test_folder", testing_dict)

    return {"num_categories":classification, "height":height, "width":width}

def add_resized_images(data_path, folder_name, images):

    # create processed folder
    os.mkdir(data_path + "/proc_" + folder_name)
    # add images to processed folder
    for img_name, img in images.items():
        cv2.imwrite(data_path + "/proc_" + folder_name + "/proc_" + img_name, img)


def replace_images(data_path, loaded_shaped):

    for img_name, img in loaded_shaped.items():
        cv2.imwrite(data_path + "/" + img_name, img)


def process_color_channel(img, height, width):
    chanels = [chanel for chanel in cv2.split(img)]

    for index, chanel in enumerate(chanels):
        if chanel.shape[0] > height:
            chanel = cv2.resize(chanel, dsize=(chanel.shape[1], height), interpolation=cv2.INTER_CUBIC)
        else:
            chanel = cv2.resize(chanel, dsize=(chanel.shape[1], height), interpolation=cv2.INTER_AREA)
        if chanel.shape[1] > width:
            chanel = cv2.resize(chanel, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        else:
            chanel = cv2.resize(chanel, dsize=(width, height), interpolation=cv2.INTER_AREA)
        chanels[index] = chanel

    return cv2.merge(chanels)


# Seperates the color channels and then reshapes each of the channels to
# (224, 224)
# def processColorChanel(img):
#     b, g, r = cv2.split(img)
#     # seperating each value into a color channel and resizing to a standard
#     # size of 224, 224, 3 <- because of RGB color channels. If it's not 3
#     # color channels it'll pad with zeroes
#     b = cv2.resize(b, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
#     g = cv2.resize(g, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
#     r = cv2.resize(r, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
#     img = cv2.merge((b, g, r))
#     return img


def process_dates(data):

    for df in data.values():
        datetime_cols = df.select_dtypes('datetime64')

        for col in datetime_cols:

            df[f'{col}_DayOfWeek'] = df[col].dt.day_name()
            df[f'{col}_Year'] = df[col].dt.year
            df[f'{col}_Month'] = df[col].dt.month_name()
            df[f'{col}_MonthDay'] = df[col].dt.day

            del df[col]
