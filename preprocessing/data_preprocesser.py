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
import cv2


# Preprocesses the data appropriately for single reg data
def structured_preprocesser(data):
    # identifies the categorical and numerical columns
    categorical_columns = data.select_dtypes(exclude=["number"]).columns
    numeric_columns = data.columns[data.dtypes.apply(
        lambda c: np.issubdtype(c, np.number))]

    # Only runs if the categorical columns is more than zero
    if(len(categorical_columns) != 0):

        # Creates a feature mask and then transforms it using a labeler and a
        # One Hot Encoder
        categorical_feature_mask = data.dtypes == object
        categorical_cols = data.columns[categorical_feature_mask].tolist()
        labeled_df = data[categorical_cols]

        labeled_df.fillna("", inplace=True)
        enc = OneHotEncoder()
        enc.fit(labeled_df)
        onehotlabels = enc.transform(labeled_df).toarray()

        new_columns = list()

        # Changes the columns into the right shape
        for col, values in zip(labeled_df.columns, enc.categories_):
            new_columns.extend([col + '_' + str(value) for value in values])

        data = pd.concat([data, pd.DataFrame(
            onehotlabels, columns=new_columns)], axis='columns')

        # Deletes the main columns that're not One Hot Encoded
        for x in categorical_cols:
            del data[x]

    if(len(numeric_columns) != 0):
        # Imputes numeric columns with median
        imputer = SimpleImputer(strategy="median")
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        # Scales numeric data
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data

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
