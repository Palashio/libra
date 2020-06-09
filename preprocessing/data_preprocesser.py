import keras
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
    # get target column
    remove = get_similar_column(
            get_value_instruction(instruction), data)
    y = data[remove]

    # remove rows where target is NaN
    data = data[y.notna()]
    y = y[y.notna()]
    del data[remove]

    #identification of id columns: if they're an unique and non-numerical we have to remove
    for column in data.columns:
        if len(np.unique(data[column])) == len(data) and data[column].dtype.name == 'object':
            del data[column]

    # preprocess the dataset
    full_pipeline = None
    if preprocess:
        data, full_pipeline = structured_preprocesser(data)
    else:
        data.fillna(0, inplace=True)

    return data, y, remove, full_pipeline


# Preprocesses the data appropriately for single reg data
def structured_preprocesser(data):
    # identifies the categorical and numerical columns
    categorical_columns = data.select_dtypes(exclude=["number"]).columns
    numeric_columns = data.columns[data.dtypes.apply(
        lambda c: np.issubdtype(c, np.number))]
    
    # pipeline for numeric columns
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    # pipeline for categorical columns
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="constant", fill_value="")),
        ('one_hot_encoder', OneHotEncoder()),
    ])

    full_pipeline = None
    # combine the two pipelines
    if(len(numeric_columns) != 0 and len(categorical_columns) != 0):
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, numeric_columns),
            ("cat", cat_pipeline, categorical_columns),
        ])
    elif len(numeric_columns) == 0:
        full_pipeline = ColumnTransformer([
            ("cat", cat_pipeline, categorical_columns),
        ])
    else:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, numeric_columns),
        ])

    data = full_pipeline.fit_transform(data)
    new_columns = list(numeric_columns)

    if len(categorical_columns) != 0:
        # create labels for resultant dataframe
        enc = full_pipeline.named_transformers_['cat']['one_hot_encoder']
        for col, values in zip(categorical_columns, enc.categories_):
                new_columns.extend([col + '_' + str(value) for value in values])
            
    return pd.DataFrame(data, columns=new_columns), full_pipeline

# Preprocesses images queried from images to (224, 224, 3)


def image_preprocess(data_path):
    image_dir = str(data_path)
    loaded_shaped = []
    imagesList = listdir(image_dir)

    for image in imagesList:
        try:
            img = cv2.imread(image_dir + "/" + image)
            res = processColorChanel(img)
            loaded_shaped.append(res)
            # print(res)
        except BaseException:
            continue

    return loaded_shaped


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
