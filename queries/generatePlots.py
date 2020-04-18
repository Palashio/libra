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


def generateClusteringPlots(kmeans, dataset):
        for x in range(len(dataset) - 1):
                plt.scatter(dataset[:,x],dataset[:,x+1], c=kmeans.labels_, cmap='rainbow')
                plt.show()
