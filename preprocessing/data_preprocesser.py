import keras
import numpy as np
import pandas as pd
import os
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
    # Scans for object columns just in case we have a datetime column that isn't detected
    object_columns = [col for col, col_type in data.dtypes.iteritems() if col_type == 'object']

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
    categorical_columns = data['train'].select_dtypes(exclude=["number"]).columns
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
    train_encoded_cols = full_pipeline.named_transformers_['cat']['one_hot_encoder'].get_feature_names()
    train_cols = [*list(numeric_columns), *train_encoded_cols]

    test = full_pipeline.transform(data['test'])
    test_encoded_cols = full_pipeline.named_transformers_['cat']['one_hot_encoder'].get_feature_names()
    test_cols = [*list(numeric_columns), *test_encoded_cols]

    # Ternary clause because when running housing.csv,
    # the product of preprocessing is np array, but not when using landslide data... not sure why
    data['train'] = pd.DataFrame((train.toarray() if not isinstance(train, np.ndarray) else train), columns=train_cols)
    data['test'] = pd.DataFrame((test.toarray() if not isinstance(train, np.ndarray) else test), columns=test_cols)

    return data, full_pipeline


# def image_preprocess(data_path):
#     image_dir = str(data_path)
#     loaded_shaped = []
#     imagesList = listdir(image_dir)
#
#     for image in imagesList:
#         try:
#             img = cv2.imread(image_dir + "/" + image)
#             res = processColorChanel(img)
#             loaded_shaped.append(res)
#             # print(res)
#         except BaseException:
#             continue
#
#     return loaded_shaped

# Preprocesses images queried from images to median of heighs/widths
def image_preprocess2(data_path, new_folder=True):
    image_dir = str(data_path)
    loaded_shaped = {}
    imagesList = listdir(image_dir)

    # store all the widths and heights of images
    heights = []
    widths = []

    # get median of widths and heights
    for image in imagesList:
        try:
            img = cv2.imread(image_dir + "/" + image)
            heights.append(img.shape[0])
            widths.append(img.shape[1])
            loaded_shaped[image] = img
            #loaded_shaped.append(img)
        except BaseException:
            continue

    heights.sort()
    widths.sort()
    height = heights[int(len(heights)/2)]
    width = widths[int(len(widths)/2)]

    # resize images
    for img_name, image in loaded_shaped.items():
        loaded_shaped[img_name] = processColorChanel2(image, height, width)

    # create new folder containing resized images
    if (new_folder):
        addResizedImages(data_path, loaded_shaped)
    else:
        replaceImages(data_path, loaded_shaped)

def addResizedImages(data_path, loaded_shaped):
    cwd = os.getcwd()

    # create processed folder
    os.chdir(data_path)
    class_name = os.path.basename(os.getcwd())
    os.chdir("..")
    os.mkdir("proc_" + class_name)
    os.chdir("proc_" + class_name)
    # add images to processed folder
    for img_name, img in loaded_shaped.items():
        cv2.imwrite("proc_" + img_name, img)

    os.chdir(cwd)

def replaceImages(data_path, loaded_shaped):
    cwd = os.getcwd()
    os.chdir(data_path)

    for img_name, img in loaded_shaped.items():
        cv2.imwrite(img_name, img)

    os.chdir(cwd)

def processColorChanel2(img, height, width):
    chanels = [chanel for chanel in cv2.split(img)]

    for index, chanel in enumerate(chanels):
        if chanel.shape[0] > height:
            chanel = cv2.resize(chanel, dsize=(chanel.shape[1], height), interpolation=cv2.INTER_CUBIC)
        else:
            chanel = cv2.resize(chanel, dsize=(chanel.shape[1], height), interpolation=cv2.INTER_AREA)
        if chanel.shape[1] > width:
            chanel = cv2.resize(chanel, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        else:
            channel = cv2.resize(chanel, dsize=(width, height), interpolation=cv2.INTER_AREA)
        chanels[index] = chanel

    return cv2.merge(chanels)


# Seperates the color channels and then reshapes each of the channels to
# (224, 224)
def processColorChanel(img):
    b, g, r = cv2.split(img)
    # seperating each value into a color channel and resizing to a standard
    # size of 224, 224, 3 <- because of RGB color channels. If it's not 3
    # color channels it'll pad with zeroes
    b = cv2.resize(b, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    g = cv2.resize(g, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    r = cv2.resize(r, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.merge((b, g, r))
    return img


def process_dates(data):

    for df in data.values():
        datetime_cols = df.select_dtypes('datetime64')

        for col in datetime_cols:

            df[f'{col}_DayOfWeek'] = df[col].dt.day_name()
            df[f'{col}_Year'] = df[col].dt.year
            df[f'{col}_Month'] = df[col].dt.month_name()
            df[f'{col}_MonthDay'] = df[col].dt.day

            del df[col]


