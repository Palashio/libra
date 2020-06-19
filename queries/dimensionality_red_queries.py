import sys

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data_generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')

from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing, tree
from itertools import product, permutations
from data_reader import DataReader
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from tuner import tuneReg, tuneClass
from os import listdir
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from generate_plots import generate_clustering_plots, generate_regression_plots, generate_classification_plots, generate_classification_together
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.utils import np_utils
from keras.utils import to_categorical
from prediction_model_creation import get_keras_model_class
from prediction_model_creation import get_keras_model_reg
from data_preprocesser import structured_preprocesser
from grammartree import get_value_instruction
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
from dataset_labelmatcher import get_similar_column
from tensorflow.python.keras.layers import Dense, Input
from xgboost import XGBClassifier
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
from sklearn.decomposition import PCA, FastICA, KernelPCA
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
# function imports from other files


currLog = ""
counter = 0

# allows for all columns to be displayed when printing()
pd.options.display.width = None


def logger(instruction, found=""):
    global currLog
    global counter

    if counter == 0:
        currLog += (" " * 2 * counter) + instruction + found + "\n"
    else:
        currLog += (" " * 2 * counter) + "|" + "\n"
        currLog += (" " * 2 * counter) + "|- " + instruction + found + "\n"
        if instruction == "done...":
            currLog += "\n" + "\n"

    counter += 1
    print(currLog)
    currLog = ""


def dimensionality_reduc(
        instruction,
        dataset,
        arr=[
            "RF",
            "PCA",
            "KPCA",
            "ICA"],
        inplace=False):
    global currLog
    global counter

    dataReader = DataReader(dataset)

    logger("loading dataset...")
    data = dataReader.data_generator()
    data.fillna(0, inplace=True)

    logger("getting most similar column from instruction...")
    target = get_similar_column(get_value_instruction(instruction), data)

    y = data[target]
    del data[target]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    data = structured_preprocesser(data)

    perms = []
    overall_storage = []
    finals = []

    logger("generating dimensionality permutations...")
    for i in range(1, len(arr) + 1):
        for elem in list(permutations(arr, i)):
            perms.append(elem)

    logger("running each possible permutation...")
    logger("realigning tensors...")
    for path in perms:
        currSet = data
        for element in path:
            if element == "RF":
                data_mod, beg_acc, final_acc, del_cols, y_pred = dimensionality_RF(
                    instruction, currSet, target, y)
            elif element == "PCA":
                data_mod, beg_acc, final_acc, del_cols, y_pred = dimensionality_PCA(
                    instruction, currSet, target, y)
            elif element == "KPCA":
                data_mod, beg_acc, final_acc, del_cols, y_pred = dimensionality_KPCA(
                    instruction, currSet, target, y)
            elif element == "ICA":
                data_mod, beg_acc, final_acc, del_cols, y_pred = dimensionality_ICA(
                    instruction, currSet, target, y)
            overall_storage.append(
                    list([data_mod, beg_acc, final_acc, y_pred]))
            currSet=data_mod
        finals.append(overall_storage[len(overall_storage) - 1])

    logger("Fetching Best Accuracies...")
    accs = []
    print("")
    print("Baseline Accuracy: " + str(finals[0][1]))
    print("----------------------------")
    for i, element in product(range(len(finals)), finals):
        print("Permutation --> " +
              str(perms[i]) +
              " | Final Accuracy --> " +
              str(element[2]))
        if finals[0][1] < element[2]:
            accs.append(list(["Permutation --> " +
                              str(perms[i]) +
                              " | Final Accuracy --> " +
                              str(element[2])]))
    print("")
    print("Best Accuracies")
    print("----------------------------")
    print(*accs, sep = "\n")

    if inplace:
        data.to_csv(dataset)

def data_y(instruction):
    dataReader = DataReader("./data/" + get_last_file()[0])
    data = dataReader.data_generator()
    data.fillna(0, inplace=True)
    remove = get_similar_column(get_value_instruction(instruction), data)
    data = structured_preprocesser(data)
    y = data[remove]
    del data[remove]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return y
def dimensionality_RF(instruction, dataset, target="", y="", n_features=10):
    global currLog
    global counter

    y=data_y(instruction) if target == "" else y

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, y, test_size=0.2, random_state=49)
    first_classifier = tree.DecisionTreeClassifier()
    first_classifier.fit(X_train, y_train)

    first_classifier_acc = accuracy_score(
        first_classifier.predict(X_test), y_test)

    accuracy_scores = [first_classifier_acc]
    columns = []
    datas = []
    datas.append(dataset)
    columns.append([])
    y_pred=[]
    for i, x in product(range(3, 10), range(4, len(dataset.columns))):
        feature_model = RandomForestRegressor(random_state=1, max_depth=i)
        feature_model.fit(X_train, y_train)

        importances = feature_model.feature_importances_
        indices = np.argsort(importances)[-x:]
        columns.append(dataset.columns[indices])

        X_temp_train = X_train[dataset.columns[indices]]
        X_temp_test = X_test[dataset.columns[indices]]

        val = pd.DataFrame(np.r_[X_temp_train, X_temp_test])
        val[target] = np.r_[y_train, y_test]
        datas.append(val)

        vr = tree.DecisionTreeClassifier()
        vr.fit(X_temp_train, y_train)
        y_pred.append(vr.predict(X_temp_test))
        accuracy_scores.append(accuracy_score(vr.predict(X_temp_test), y_test))

    the_index = accuracy_scores.index(max(accuracy_scores))

    return datas[the_index], accuracy_scores[0], max(
        accuracy_scores), list(columns[the_index]), y_pred[the_index]


def dimensionality_PCA(instruction, dataset, target="", y=""):
    global currLog
    global counter

    y=data_y(instruction) if target == "" else y
    
    #  PCA will hold 92% of the variance
    pca = PCA(0.92)
    #pca = PCA(n_components=len(dataset.columns))
    data_modified = pca.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, y, test_size=0.2, random_state=49)
    X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(
        data_modified, y, test_size=0.2, random_state=49)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    clf_mod = tree.DecisionTreeClassifier()
    clf_mod.fit(X_train_mod, y_train_mod)
    acc=accuracy_score(
            clf_mod.predict(X_test_mod), y_test_mod)
    y_pred=clf_mod.predict(X_test_mod)
    for i,j in product(range(3,10), ["entropy","gini"]):
            model=tree.DecisionTreeClassifier(criterion=j, max_depth=i)
            model=model.fit(X_train_mod,y_train_mod)
            if accuracy_score(model.predict(X_test_mod)
                    ,y_test)>acc:
                acc=accuracy_score(model.predict(X_test_mod)
                    ,y_test)
                y_pred=model.predict(X_test_mod)
    del i,j
    data_modified = pd.DataFrame(data_modified)
    data_modified[target] = np.r_[y_train, y_test]
    # data_modified.to_csv("./data/housingPCA.csv")

    return data_modified, accuracy_score(
            clf.predict(X_test), y_test), acc, (len(
            dataset.columns) - len(data_modified.columns)), y_pred


def dimensionality_ICA(instruction, dataset, target="", y=""):
    global currLog
    global counter

    if target == "":
        y=data_y(instruction)

    pca = FastICA(n_components=len(dataset.columns))
    data_modified = pca.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, y, test_size=0.2, random_state=49)
    X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(
        data_modified, y, test_size=0.2, random_state=49)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    clf_mod = tree.DecisionTreeClassifier()
    clf_mod.fit(X_train_mod, y_train_mod)
    acc=accuracy_score(
            clf_mod.predict(X_test_mod), y_test_mod)
    y_pred=clf_mod.predict(X_test_mod)
    for i,j in product(range(3,10), ["entropy","gini"]):
            model=tree.DecisionTreeClassifier(criterion=j, max_depth=i)
            model=model.fit(X_train,y_train)
            if accuracy_score(model.predict(X_test_mod)
                    ,y_test)>acc:
                acc=accuracy_score(model.predict(X_test_mod)
                    ,y_test)
                y_pred=model.predict(X_test_mod)
    del i,j
    data_modified = pd.DataFrame(data_modified)
    data_modified[target] = np.r_[y_train, y_test]
    # data_modified.to_csv("./data/housingPCA.csv")
    
    return data_modified, accuracy_score(
            clf.predict(X_test), y_test), acc, (len(
            dataset.columns) - len(data_modified.columns)), y_pred


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


def dimensionality_KPCA(instruction, dataset, target="", y=""):
    global currLog
    global counter

    if target == "":
        y=data_y(instruction)
    
    kpca = KernelPCA(n_components=len(dataset.columns),kernel="rbf")
    data_modified = kpca.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, y, test_size=0.2, random_state=49)
    X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(
        data_modified, y, test_size=0.2, random_state=49)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    clf_mod = tree.DecisionTreeClassifier()
    clf_mod.fit(X_train_mod, y_train_mod)
    acc=accuracy_score(
            clf_mod.predict(X_test_mod), y_test_mod)
    y_pred=clf_mod.predict(X_test_mod)
    for i,j in product(range(3,10), ["entropy","gini"]):
            model=tree.DecisionTreeClassifier(criterion=j, max_depth=i)
            model=model.fit(X_train_mod,y_train_mod)
            if accuracy_score(model.predict(X_test_mod)
                    ,y_test)>acc:
                acc=accuracy_score(model.predict(X_test_mod)
                    ,y_test)
                y_pred=model.predict(X_test_mod)
    del i,j
    data_modified = pd.DataFrame(data_modified)
    data_modified[target] = np.r_[y_train, y_test]
    # data_modified.to_csv("./data/housingPCA.csv")

    return data_modified, accuracy_score(
            clf.predict(X_test), y_test), acc, (len(
            dataset.columns) - len(data_modified.columns)), y_pred


dimensionality_PCA("Predict median house value", "./data/housing.csv")

#dimensionalityReduc("Predict ocean_proximity", "./data/housing.csv")

# data = pd.read_csv("./data/housing.csv")
# data.fillna(0, inplace=True)
# target = get_similar_column(get_value_instruction("Predict ocean proximity"), data)

# y = data[target]
# del data[target]
# le = preprocessing.LabelEncoder()
# y = le.fit_transform(y)

# data = structured_preprocesser(data)

#dimensionalityICA("Predict ocean proximity", data, "ocean_proximity", y)
