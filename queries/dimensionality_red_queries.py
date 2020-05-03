from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing, tree
import itertools
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from tuner import tuneReg, tuneClass
from os import listdir
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from dataGen import generate_data
from generatePlots import generateClusteringPlots, generateRegressionPlots, generateClassificationPlots, generateClassificationTogether
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.utils import np_utils
from keras.utils import to_categorical
from predictionModelCreation import getKerasModelClassification
from predictionModelCreation import getKerasModelRegression
from data_preprocesser import singleRegDataPreprocesser, preProcessImages
from grammartree import getValueFromInstruction
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
from dataset_labelmatcher import getmostSimilarColumn
from tensorflow.python.keras.layers import Dense, Input
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from scipy.spatial.distance import cosine
from tabulate import tabulate
from sklearn.decomposition import PCA, FastICA
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import sys

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')

# function imports from other files

currLog = ""
counter = 0

# allows for all columns to be displayed when printing()
pd.options.display.width = None

# same logger as predictionQueries


def logger(instruction, found=""):
    global currLog
    global counter

    if counter == 0:
        currLog += (" " * 2 * counter) + instruction + found
        currLog += "\n"
    else:
        currLog += (" " * 2 * counter) + "|"
        currLog += "\n"
        currLog += (" " * 2 * counter) + "|- " + instruction + found
        currLog += "\n"
        if instruction == "done...":
            currLog += "\n"
            currLog += "\n"

    counter += 1
    print(currLog)


# this is the pipeliner for dimensionalityReduc()
def dimensionalityReduc(instruction, dataset):
    global currLog
    global counter

    # loads dataset and preprocesses it initially to smooth out pipelining
    logger("loading dataset...")
    data = pd.read_csv(dataset)
    data.fillna(0, inplace=True)

    # get most similar column by using the instruction
    logger("getting most similar column from instruction...")
    target = getmostSimilarColumn(getValueFromInstruction(instruction), data)

    # encodes dataset into labels for preprocessing and isolates it (this
    # isn't modified in the pipeline)
    y = data[target]
    del data[target]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    data = singleRegDataPreprocesser(data)

    # different dimensionality reduction techniques it supports
    perms = []
    overall_storage = []
    finals = []
    arr = ["RF", "PCA", "ICA"]

    # generating all possible permutations for dimensionality reduction
    # techniques
    logger("generating dimensionality permutations...")
    for i in range(1, len(arr) + 1):
        for elem in list(itertools.permutations(arr, i)):
            perms.append(elem)

    logger("running each possible permutation...")
    logger("realigning tensors...")

    # identifies every possible path and returns the accuracy for each
    # permutation/path
    for path in perms:
        storage = []
        storage.append(data)
        for element in path:
            # Each of these if blocks are for a different pipeline, it's run and stored in the storage [] array. Each one takes the last dataset in storage and runs it through
            # the reduction pipeline.
            if element == "RF":
                currSet = storage[len(storage) - 1]
                data_mod, beg_acc_RF, final_acc_RF, col_removed_RF = dimensionalityRF(
                    instruction, currSet, target, y)
                storage.append(data_mod)
                overall_storage.append(
                    list([data_mod, beg_acc_RF, final_acc_RF, col_removed_RF]))
            if element == "PCA":
                currSet = storage[len(storage) - 1]
                data_mod, beg_acc_PCA, final_acc_PCA, col_removed_PCA = dimensionalityPCA(
                    instruction, currSet, target, y)
                storage.append(data_mod)
                overall_storage.append(
                    list([data_mod, beg_acc_PCA, final_acc_PCA, col_removed_PCA]))
            if element == "ICA":
                currSet = storage[len(storage) - 1]
                data_mod, beg_acc_ICA, final_acc_ICA, col_removed_ICA = dimensionalityICA(
                    instruction, currSet, target, y)
                storage.append(data_mod)
                overall_storage.append(
                    list([data_mod, beg_acc_ICA, final_acc_ICA, col_removed_ICA]))
            # If we're at the end of that specific pipeline permutations we
            # want to move onto the next one
            if path.index(element) == len(path) - 1:
                finals.append(overall_storage[len(overall_storage) - 1])

    logger("getting best accuracies...")

    # printing out all of the accuracies appropriately
    accs = []
    i = 0
    print("")
    print("Baseline Accuracy: " + str(finals[0][1]))
    print("----------------------------")
    for element in finals:
        print("Permutation --> " +
              str(perms[i]) +
              " | Final Accuracy --> " +
              str(element[2]))
        if finals[0][1] < element[2]:
            accs.append(list(["Permutation --> " +
                              str(perms[i]) +
                              " | Final Accuracy --> " +
                              str(element[2])]))
        i += 1
    print("")
    print("Best Accuracies")
    print("----------------------------")
    for element in accs:
        print(element)


def dimensionalityRF(instruction, dataset, target="", y=""):
    global currLog
    global counter

    # identifies where the user called this specific reduciton technique or it
    # was accessed through the whole pipeline
    if target == "":
        data = pd.read_csv("./data/" + get_last_file()[0])
        data.fillna(0, inplace=True)
        remove = getmostSimilarColumn(
            getValueFromInstruction(instruction), data)
        data = singleRegDataPreprocesser(data)

        y = data[remove]
        del data[remove]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

    # if pipelined then we want to skip over processing and go directly into
    # splitting for training/testing
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, y, test_size=0.2, random_state=49)
    first_classifier = tree.DecisionTreeClassifier()
    first_classifier.fit(X_train, y_train)

    # base accuracy we want to compare to
    first_classifier_acc = accuracy_score(
        first_classifier.predict(X_test), y_test)

    # storages, and all initial appending for run
    accuracy_scores = [first_classifier_acc]
    columns = []
    datas = []
    datas.append(dataset)
    columns.append([])

    # Iterates through all possible Random Forest selectors (either
    # 4-len(columns) of features that you want to keep and stores them)
    for x in range(4, len(dataset.columns)):
        feature_model = RandomForestRegressor(random_state=1, max_depth=10)
        feature_model.fit(X_train, y_train)

        importances = feature_model.feature_importances_
        indices = np.argsort(importances)[-x:]
        columns.append(dataset.columns[indices])

        X_temp_train = X_train[dataset.columns[indices]]
        X_temp_test = X_test[dataset.columns[indices]]

        # re-indexes and stores into dataset using np indexing
        X_combined = np.r_[X_temp_train, X_temp_test]
        y_combined = np.r_[y_train, y_test]
        val = pd.DataFrame(X_combined)
        val[target] = y_combined
        datas.append(val)

        vr = tree.DecisionTreeClassifier()
        vr.fit(X_temp_train, y_train)

        accuracy_scores.append(accuracy_score(vr.predict(X_temp_test), y_test))

    the_index = accuracy_scores.index(max(accuracy_scores))

    # returns best dataset, the beginning accuracy, the final accuracy, and
    # the columns removed
    return datas[the_index], accuracy_scores[0], max(
        accuracy_scores), list(columns[the_index])


def dimensionalityPCA(instruction, dataset, target="", y="", n_components=10):
    global currLog
    global counter

    # identifies where the user called this specific reduciton technique or it
    # was accessed through the whole pipeline
    if target == "":
        data = pd.read_csv("./data/" + get_last_file()[0])
        data.fillna(0, inplace=True)
        remove = getmostSimilarColumn(
            getValueFromInstruction(instruction), data)

        y = data[remove]
        del data[remove]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

    # principle component analysis
    pca = PCA(n_components=len(dataset.columns))
    data_modified = pca.fit_transform(dataset)

    # splitting of sets
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, y, test_size=0.2, random_state=49)
    X_train_mod, none, y_train_mod, none1 = train_test_split(
        data_modified, y, test_size=0.2, random_state=49)

    # creating both initial decision tree and final decision tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    clf_mod = tree.DecisionTreeClassifier()
    clf_mod.fit(X_train_mod, y_train_mod)

    accuracies = [
        accuracy_score(
            clf.predict(X_test), y_test), accuracy_score(
            clf_mod.predict(none), none1)]
    data_modified = pd.DataFrame(data_modified)

    # re-combines into dataset and adds back to data_modified
    y_combined = np.r_[y_train, y_test]
    data_modified[target] = y_combined
    # data_modified.to_csv("./data/housingPCA.csv")

    # returns best dataset, the beginning accuracy, the final accuracy, and
    # the columns removed
    return data_modified, accuracies[0], accuracies[1], (len(
        dataset.columns) - len(data_modified.columns))


def dimensionalityICA(instruction, dataset, target="", y=""):
    global currLog
    global counter

    # identifies where the user called this specific reduciton technique or it
    # was accessed through the whole pipeline
    if target == "":
        data = pd.read_csv("./data/" + get_last_file()[0])
        data.fillna(0, inplace=True)
        remove = getmostSimilarColumn(
            getValueFromInstruction(instruction), data)

        y = data[remove]
        del data[remove]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

    # independent component analysis identification
    pca = FastICA(n_components=len(dataset.columns))
    data_modified = pca.fit_transform(dataset)

    # splitting both the original and modified sets into their appropriate
    # dataset sizes
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, y, test_size=0.2, random_state=49)
    X_train_mod, none, y_train_mod, none1 = train_test_split(
        data_modified, y, test_size=0.2, random_state=49)

    # classifier to test each individual accuracy difference
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    clf_mod = tree.DecisionTreeClassifier()
    clf_mod.fit(X_train_mod, y_train_mod)

    accuracies = [
        accuracy_score(
            clf.predict(X_test), y_test), accuracy_score(
            clf_mod.predict(none), none1)]
    data_modified = pd.DataFrame(data_modified)

    y_combined = np.r_[y_train, y_test]
    data_modified[target] = y_combined
    # data_modified.to_csv("./data/housingPCA.csv")

    # returns best dataset, the beginning accuracy, the final accuracy, and
    # the number of columns that were removed
    return data_modified, accuracies[0], accuracies[1], (len(
        dataset.columns) - len(data_modified.columns))


# gets the newest file in the ./data directory. This isn't used but is a
# helper function previously written before functionality changes.
def get_last_file():
    max_mtime = 0
    for dirname, subdirs, files in os.walk("./data"):
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
