import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
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
    data.fillna(0, inplace=True)

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
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data
