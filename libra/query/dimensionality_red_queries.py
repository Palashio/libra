from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing, tree
from itertools import product, permutations
from libra.preprocessing.data_reader import DataReader
import os
from sklearn.ensemble import RandomForestRegressor
from libra.preprocessing.data_preprocesser import structured_preprocesser, initial_preprocesser, clustering_preprocessor
from libra.data_generation.grammartree import get_value_instruction
from libra.data_generation.dataset_labelmatcher import get_similar_column

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, FastICA, KernelPCA
import pandas as pd
import numpy as np
from colorama import Fore, Style
# function imports from other files

counter = 0

# allows for all columns to be displayed when printing()
pd.options.display.width = None


def logger(instruction, found=""):
    global counter
    if counter == 0:
        print((" " * 2 * counter) + str(instruction) + str(found))
    elif instruction == "->":
        counter = counter - 1
        print(Fore.BLUE + (" " * 2 * counter) +
              str(instruction) + str(found) + (Style.RESET_ALL))
    else:
        print((" " * 2 * counter) + "|- " + str(instruction) + str(found))
        if instruction == "done...":
            print("\n" + "\n")

    counter += 1


def printtable(col_name, col_width):
    global counter
    for row in col_name:
        print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")


def dimensionality_RF(instruction, dataset, target="", y="", n_features=10):

    global counter

    dataReader = DataReader(dataset)

    if target == "":
        data = dataReader.data_generator()
        data.fillna(0, inplace=True)
        remove = get_similar_column(get_value_instruction(instruction), data)
        data, y, target, full_pipeline = initial_preprocesser(
            data, instruction, True, 0.2, [], 0.2, random_state=49)

        le = preprocessing.LabelEncoder()
        X_train = data['train']
        y_train = y['train']
        X_test = data['test']
        y_test = y['test']

        y_train= le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)

    first_classifier = tree.DecisionTreeClassifier()
    first_classifier.fit(X_train, y_train)

    first_classifier_acc = accuracy_score(
        first_classifier.predict(X_test), y_test)

    accuracy_scores = [first_classifier_acc]
    columns = []
    datas = []
    datas.append(dataset)
    columns.append([])

    for i, x in product(range(3, 10), range(4, len(X_train.columns))):
        feature_model = RandomForestRegressor(random_state=1, max_depth=x)
        feature_model.fit(X_train, y_train)

        importances = feature_model.feature_importances_
        indices = np.argsort(importances)[-x:]
        columns.append(X_train.columns[indices])

        X_temp_train = X_train[X_train.columns[indices]]
        X_temp_test = X_test[X_train.columns[indices]]

        val = pd.DataFrame(np.r_[X_temp_train, X_temp_test])
        val[target] = np.r_[y_train, y_test]
        datas.append(val)

        vr = tree.DecisionTreeClassifier()
        vr.fit(X_temp_train, y_train)

        accuracy_scores.append(accuracy_score(vr.predict(X_temp_test), y_test))

    the_index = accuracy_scores.index(max(accuracy_scores))
    print(accuracy_scores)
    return datas[the_index], accuracy_scores[0], max(
        accuracy_scores), list(columns[the_index])


def dimensionality_PCA(instruction, dataset, ca_threshold=None):

    global counter

    pca = PCA(0.92)

    dataReader = DataReader(dataset)
    dataset = dataReader.data_generator()

    data, y, target, full_pipeline = initial_preprocesser(
        dataset, instruction, True, 0.2, [], 0.2, random_state=49)


    X_train = data['train']
    X_test = data['test']

    y_train = y['train']
    y_test = y['test']

    X_train_mod = pca.fit_transform(X_train)
    X_test_mod = pca.transform(X_test)

    clf = tree.DecisionTreeClassifier()
    clf_mod = tree.DecisionTreeClassifier()

    clf.fit(X_train, y_train)
    clf_mod.fit(X_train_mod, y_train)

    acc = []
    acc.append(accuracy_score(
        clf_mod.predict(X_test_mod), y_test))

    for i, j in product(range(3, 10), ["entropy", "gini"]):
        model = tree.DecisionTreeClassifier(criterion=j, max_depth=i)
        model = model.fit(X_train_mod, y_train)
        acc.append(accuracy_score(model.predict(X_test_mod), y_test))
    del i, j

    data_modified = pd.concat(
        [pd.DataFrame(X_train_mod), pd.DataFrame(X_test_mod)], axis=0)

    y_combined = np.r_[y_train, y_test]
    data_modified[target] = y_combined
    # data_modified.to_csv("./data/housingPCA.csv")

    return data_modified, accuracy_score(
        clf.predict(X_test), y_test), max(acc), (len(
            dataset.columns) - len(data_modified.columns))


def dimensionality_ICA(instruction, dataset, target="", y=""):

    global counter

    dataReader = DataReader(dataset)

    if target == "":
        data = dataReader.data_generator()
        data.fillna(0, inplace=True)
        remove = get_similar_column(get_value_instruction(instruction), data)

        data, y, target, full_pipeline = initial_preprocesser(
            data, instruction, True, 0.2, [], 0.2, random_state=49)

        X_train = data['train']
        X_test = data['test']

        y_train = y['train']
        y_test = y['test']


    pca = FastICA(n_components=len(X_train.columns))
    X_train_mod = pca.fit_transform(X_train)
    X_test_mod = pca.fit_transform(X_test)


    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    clf_mod = tree.DecisionTreeClassifier()
    clf_mod.fit(X_train_mod, y_train)
    acc = []
    sets = []
    acc.append(accuracy_score(
        clf_mod.predict(X_test_mod), y_test))

    frame = pd.DataFrame(pd.DataFrame(X_train_mod).append(pd.DataFrame(X_test_mod)))
    frame[target] = np.r_[y_train, y_test]
    sets.append(frame)

    for i in range(2, len(X_train.columns)):
        pca = FastICA(n_components=i)
        X_train_mod = pca.fit_transform(X_train)
        X_test_mod = pca.fit_transform(X_test)

        frame = pd.DataFrame(pd.DataFrame(X_train_mod).append(pd.DataFrame(X_test_mod)))
        frame[target] = np.r_[y_train, y_test]
        sets.append(frame)

        clf_mod = tree.DecisionTreeClassifier()
        clf_mod.fit(X_train_mod, y_train)

        acc.append(accuracy_score(
            clf_mod.predict(X_test_mod), y_test))

    del i

    data_modified = sets[acc.index(max(acc))]
    score = max(acc)


    return data_modified, score, ((len(
            X_train.columns) + 1) - len(data_modified.columns))


def get_last_file():
    '''
    function to retrieve most recently created file generated from
    reduced dimensionality technique
    '''
    
    max_mtime = 0
    max_file = ""
    max_dir = ""
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
    '''
    function to reduce dimensionality in dataset via kernal principal component analysis
    :param instruction: command sent to client instance in written query.
    :param dataset: data instantiated in client instance passed to the algorithm
    :param target: column name of response variable/feature
    :param y: dictionary of train/test data values associated with response variable/feature
    '''
    
    pca = KernelPCA(kernel='rbf')

    dataReader = DataReader(dataset)
    dataset = dataReader.data_generator()

    data, y, target, full_pipeline = initial_preprocesser(
        dataset, instruction, True, 0.2, [], 0.2, random_state=49)


    X_train = data['train']
    X_test = data['test']

    y_train = y['train']
    y_test = y['test']

    X_train_mod = pca.fit_transform(X_train)
    X_test_mod = pca.transform(X_test)

    clf = tree.DecisionTreeClassifier()
    clf_mod = tree.DecisionTreeClassifier()

    clf.fit(X_train, y_train)
    clf_mod.fit(X_train_mod, y_train)

    acc = []
    acc.append(accuracy_score(
        clf_mod.predict(X_test_mod), y_test))

    for i, j in product(range(3, 10), ["entropy", "gini"]):
        model = tree.DecisionTreeClassifier(criterion=j, max_depth=i)
        model = model.fit(X_train_mod, y_train)
        acc.append(accuracy_score(model.predict(X_test_mod), y_test))
    del i, j

    data_modified = pd.concat(
        [pd.DataFrame(X_train_mod), pd.DataFrame(X_test_mod)], axis=0)

    y_combined = np.r_[y_train, y_test]
    data_modified[target] = y_combined
    # data_modified.to_csv("./data/housingPCA.csv")

    return data_modified, accuracy_score(
        clf.predict(X_test), y_test), max(acc), (len(
            dataset.columns) - len(data_modified.columns))

