import sys

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')

#function imports from other files
import keras
import numpy as np
import itertools 
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA, FastICA
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
import os

def dimensionalityReduc(instruction, dataset):

    data = pd.read_csv(dataset)
    data.fillna(0, inplace=True)
    target = getmostSimilarColumn(getValueFromInstruction(instruction), data)

    y = data[target]
    del data[target]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    data = singleRegDataPreprocesser(data)

    perms = []
    overall_storage = []
    finals = []
    arr = ["RF", "PCA", "ICA"]

    for i in range(1, len(arr) + 1):
        for elem in list(itertools.permutations(arr, i)):
            perms.append(elem)


    for path in perms:
        storage = []
        storage.append(data)
        for element in path:
            if element == "RF":
                currSet = storage[len(storage) - 1]
                data_mod, beg_acc_RF, final_acc_RF, col_removed_RF = dimensionalityRF(instruction, currSet, target, y)
                storage.append(data_mod)
                overall_storage.append(list([data_mod, beg_acc_RF,final_acc_RF, col_removed_RF]))
            if element == "PCA":
                currSet = storage[len(storage) - 1] 
                data_mod, beg_acc_PCA, final_acc_PCA, col_removed_PCA = dimensionalityPCA(instruction, currSet, target, y)
                storage.append(data_mod)
                overall_storage.append(list([data_mod, beg_acc_PCA, final_acc_PCA, col_removed_PCA]))
            if element == "ICA":
                currSet = storage[len(storage) - 1] 
                data_mod, beg_acc_ICA, final_acc_ICA, col_removed_ICA = dimensionalityICA(instruction, currSet, target, y)
                storage.append(data_mod)
                overall_storage.append(list([data_mod, beg_acc_ICA, final_acc_ICA, col_removed_ICA]))
            if path.index(element) == len(path) - 1:
                finals.append(overall_storage[len(overall_storage) - 1])

    accs = []
    i = 0
    print("")
    print("Baseline Accuracy: " + str(finals[0][1]))
    print("----------------------------")
    for element in finals:
        print("Permutation --> " + str(perms[i]) + " | Final Accuracy --> " + str(element[2]))
        if finals[0][1] < element[2]:
            accs.append(list(["Permutation --> " + str(perms[i]) + " | Final Accuracy --> " + str(element[2])]))
        i += 1
    print("")
    print("Best Accuracies")
    print("----------------------------")
    for element in accs:
        print(element)


    



def dimensionalityRF(instruction, dataset, target="", y = ""):
    print("Initial shape: " + str(dataset.shape))
    if target == "":
        data = pd.read_csv("./data/" + get_last_file()[0])
        data.fillna(0, inplace=True)
        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
        data = singleRegDataPreprocesser(data)

        y = data[remove]
        del data[remove]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2, random_state=49)
    first_classifier = tree.DecisionTreeClassifier()
    first_classifier.fit(X_train, y_train)

    first_classifier_acc = accuracy_score(first_classifier.predict(X_test), y_test)
    
    accuracy_scores = [first_classifier_acc]
    columns = []
    datas = []
    datas.append(dataset)
    columns.append([])

    for x in range(4, len(dataset.columns)):
        feature_model = RandomForestRegressor(random_state=1, max_depth=10)
        feature_model.fit(X_train, y_train)

        importances = feature_model.feature_importances_
        indices = np.argsort(importances)[-x:]
        columns.append(dataset.columns[indices])

        X_temp_train = X_train[dataset.columns[indices]]
        X_temp_test = X_test[dataset.columns[indices]]

        X_combined = np.r_[X_temp_train, X_temp_test]
        y_combined = np.r_[y_train, y_test]
        val = pd.DataFrame(X_combined)
        val[target] = y_combined
        datas.append(val)
            
        vr = tree.DecisionTreeClassifier()
        vr.fit(X_temp_train, y_train)

        accuracy_scores.append(accuracy_score(vr.predict(X_temp_test), y_test))
    
    the_index = accuracy_scores.index(max(accuracy_scores))

    return datas[the_index], accuracy_scores[0], max(accuracy_scores), list(columns[the_index])

def dimensionalityPCA(instruction, dataset, target="", y = ""):
    if target == "":
        data = pd.read_csv("./data/" + get_last_file()[0])
        data.fillna(0, inplace=True)
        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)

        y = data[remove]
        del data[remove]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)


    pca = PCA(n_components = len(dataset.columns))
    data_modified = pca.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2, random_state=49)
    X_train_mod, none, y_train_mod, none1 = train_test_split(data_modified, y, test_size=0.2, random_state=49)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    clf_mod = tree.DecisionTreeClassifier()
    clf_mod.fit(X_train_mod, y_train_mod)

    accuracies = [accuracy_score(clf.predict(X_test), y_test), accuracy_score(clf_mod.predict(none), none1)]
    data_modified = pd.DataFrame(data_modified)
    
    y_combined = np.r_[y_train, y_test]
    data_modified[target] = y_combined
    #data_modified.to_csv("./data/housingPCA.csv")
    
    return data_modified, accuracies[0], accuracies[1], (len(dataset.columns) - len(data_modified.columns))

def dimensionalityICA(instruction, dataset, target="", y = ""):
    print(dataset.shape)
    if target == "":
        data = pd.read_csv("./data/" + get_last_file()[0])
        data.fillna(0, inplace=True)
        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)

        y = data[remove]
        del data[remove]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)


    pca = FastICA(n_components = len(dataset.columns))
    data_modified = pca.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2, random_state=49)
    X_train_mod, none, y_train_mod, none1 = train_test_split(data_modified, y, test_size=0.2, random_state=49)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    clf_mod = tree.DecisionTreeClassifier()
    clf_mod.fit(X_train_mod, y_train_mod)

    accuracies = [accuracy_score(clf.predict(X_test), y_test), accuracy_score(clf_mod.predict(none), none1)]
    data_modified = pd.DataFrame(data_modified)
    
    y_combined = np.r_[y_train, y_test]
    data_modified[target] = y_combined
    #data_modified.to_csv("./data/housingPCA.csv")
    
    print(data_modified.shape)
    return data_modified, accuracies[0], accuracies[1], (len(dataset.columns) - len(data_modified.columns))

    
def get_last_file():
    max_mtime = 0
    for dirname,subdirs,files in os.walk("./data"):
        for fname in files:
            if fname == ".DS_Store":
                continue
            full_path = os.path.join(dirname, fname)
            mtime = os.stat(full_path).st_mtime
            if mtime > max_mtime:
                max_mtime = mtime
                max_dir = dirname
                max_file = fname
    return max_file, max_dir, max_mtime

#dimensionalityPCA("Predict median house value", "./data/housing.csv")

#dimensionalityReduc("Predict ocean_proximity", "./data/housing.csv")

# data = pd.read_csv("./data/housing.csv")
# data.fillna(0, inplace=True)
# target = getmostSimilarColumn(getValueFromInstruction("Predict ocean proximity"), data)

# y = data[target]
# del data[target]
# le = preprocessing.LabelEncoder()
# y = le.fit_transform(y)

# data = singleRegDataPreprocesser(data)

dimensionalityReduc("Predict ocean proximity", "./data/housing.csv")
#dimensionalityICA("Predict ocean proximity", data, "ocean_proximity", y)