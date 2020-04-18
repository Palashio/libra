import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from data_preprocesser import singleRegDataPreprocesser
from predictionModelCreation import getKerasModelRegression
from predictionModelCreation import getKerasModelClassification
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from generatePlots import generateClusteringPlots

pd.set_option('display.max_columns', None)

def SingleRegressionQueryANN(dataset_path, instruction):
        data = pd.read_csv(dataset_path)
        data.fillna(0, inplace=True)
        
        categorical_columns = data.select_dtypes(exclude=["number"]).columns
        numeric_columns = data.columns[data.dtypes.apply(lambda c: np.issubdtype(c, np.number))]

        data = singleRegDataPreprocesser(data)
        y = data[str(instruction)]
        del data[str(instruction)]

        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

        models=[]
        losses = []
        epochs = 5

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        i = 0
        model = getKerasModelRegression(data, i)

        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
        models.append(history)

        losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])


        while(all(x > y for x, y in zip(losses, losses[1:]))):
             model = getKerasModelRegression(data, i)
             history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
             models.append(history)
             losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])
             print("The number of layers " + str(len(model.layers)))
             i += 1

        print(data)

        return models[len(models) - 1]


def classificationQueryANN(dataset_path, instruction):
    data = pd.read_csv(dataset_path)
    data.fillna(0, inplace=True)

    y = data[str(instruction)]
    del data[str(instruction)]

    data = singleRegDataPreprocesser(data)
    #classification_column = getmostSimilarColumn(getLabelwithInstruction(instruction), data)

    num_classes = len(np.unique(y))

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    y = np_utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

    models=[]
    losses = []
    epochs = 5
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    i = 0
    model = getKerasModelClassification(data, i, num_classes)

    history = model.fit(data, y, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
    models.append(history)

    losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])


    while(all(x > y for x, y in zip(losses, losses[1:]))):
        model = getKerasModelClassification(data, i, num_classes)
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
        models.append(history)
        losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])
        print("The number of layers " + str(len(model.layers)))
        i += 1
    
    return model

def kMeansClusteringQuery(dataset_path, instruction):
    data = pd.read_csv(dataset_path)
    data.fillna(0, inplace=True)
    data = np.asarray(singleRegDataPreprocesser(data))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    generateClusteringPlots(kmeans, data)





kMeansClusteringQuery("./data/housing.csv", "ocean_proximity")
