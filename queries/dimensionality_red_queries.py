import sys

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')

#function imports from other files
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from tabulate import tabulate
from scipy.spatial.distance import cosine
from pandas import DataFrame
from sklearn import preprocessing, tree 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, svm
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Input
from dataset_labelmatcher import getmostSimilarColumn
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from grammartree import getValueFromInstruction
from data_preprocesser import singleRegDataPreprocesser, preProcessImages
from predictionModelCreation import getKerasModelRegression
from predictionModelCreation import getKerasModelClassification
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from generatePlots import generateClusteringPlots, generateRegressionPlots, generateClassificationPlots, generateClassificationTogether
from dataGen import generate_data
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from os import listdir
from tuner import tuneReg, tuneClass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel    
    
def dimensionalityRF(instruction, clf, dataset, depth_search = 5, inplace = False):
    data = pd.read_csv(dataset)
    data.fillna(0, inplace=True)

    remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
    y = data[remove]
    del data[remove]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    data = singleRegDataPreprocesser(data)

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

    first_classifier = accuracy_score(clf.predict(X_test), y_test)
    
    accuracy_scores = [first_classifier]
    columns = []

    for x in range(1, depth_search):
        feature_model = RandomForestRegressor(random_state=1, max_depth=10)
        feature_model.fit(X_train, y_train)

        importances = feature_model.feature_importances_
        indices = np.argsort(importances)[-x:]
        columns.append(data.columns[indices])

        X_temp_train = X_train[data.columns[indices]]
        X_temp_test = X_test[data.columns[indices]]

        # for value in data.columns[indices]:
        #     del X_temp_train[value]
        #     del X_temp_test[value]
            
        vr = svm.SVC()
        vr.fit(X_temp_train, y_train)

        accuracy_scores.append(accuracy_score(vr.predict(X_temp_test), y_test))
    
    the_index = accuracy_scores.index(max(accuracy_scores))

    print("Accuracy before RF Regressor: " + str(accuracy_scores[0]))
    print("----------------------------------------")
    print("Best accuracy " + str(accuracy_scores[the_index]) + " by removing columns: ")
    print(list(columns[the_index]))


    #accuracy_scores.index(max(accuracy_scores))
    #print(columns)



def dimensionalityPCA(instruction, dataset):
    data = pd.read_csv(dataset)
    data.fillna(0, inplace=True)

    remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
    y = data[remove]
    del data[remove]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    data = singleRegDataPreprocesser(data)


    pca = PCA(n_components='mle')
    data_modified = pca.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)
    X_train_mod, none, y_train_mod, none1 = train_test_split(data_modified, y, test_size=0.2, random_state=49)

    print(y_train_mod.shape)
    print(X_train_mod.shape)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    clf_mod = tree.DecisionTreeClassifier()
    clf_mod.fit(X_train_mod, y_train_mod)

    accuracies = [accuracy_score(clf.predict(X_test), y_test), accuracy_score(clf_mod.predict(none), none1)]
    if accuracies.index(max(accuracies)) == 0:
        print("Principle Component Analysis should be not be used for this dataset")
        print()
        print(" -------------------------------------------------------------")
        print("| Running Feature Importance Random Forest Regressor........  |")
        print("---------------------------------------------------------------")
        dimensionalityRF(instruction, clf, dataset)

    else:
        print("Principle Component Analysis improves performance by" + str(accuracies[0] - accuracies[1]))
    


dimensionalityPCA("Predict median house value", "./data/housing.csv")
#dimensionalityRedQuery("Predict ocean_proximity", "./data/housing.csv")
