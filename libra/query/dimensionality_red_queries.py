from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing, tree
from itertools import product, permutations
from libra.preprocessing.data_reader import DataReader
import os
from sklearn.ensemble import RandomForestRegressor
from libra.preprocessing.data_preprocesser import structured_preprocesser, initial_preprocesser
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


def dimensionality_reduc(
        instruction,
        dataset,
        arr=[
            "RF",
            "PCA",
            "KPCA",
            "ICA"],
        inplace=False):
    '''
    function to perform dimensionality reduction on the dataset (retrieve only 
    features with most relevance from multidimensional space of the dataset)
    :param instruction: command sent to client instance in written query
    :param dataset: data instantiated in client instance passed to the algorithm
    :param arr: list of options of algorithm/dimension reducing techniques 
    options to choose from
    :param inplace: option to keep features that were deemed as not important
    intact in the dataset
    '''
    
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
                data_mod, beg_acc, final_acc, col_removed = dimensionality_RF(
                    instruction, currSet, target, y)
            elif element == "PCA":
                data_mod, beg_acc, final_acc, col_removed = dimensionality_PCA(
                    instruction, currSet, target, y)
            elif element == "KPCA":
                data_mod, beg_acc, final_acc, col_removed = dimensionality_KPCA(
                    instruction, currSet, target, y)
            elif element == "ICA":
                data_mod, beg_acc, final_acc, col_removed = dimensionality_ICA(
                    instruction, currSet, target, y)
            overall_storage.append(
                list([data_mod, beg_acc, final_acc, col_removed]))
            currSet = data_mod
        finals.append(overall_storage[len(overall_storage) - 1])

    logger("Fetching Best Accuracies...")
    accs = []
    logger("->", "Baseline Accuracy: " + str(finals[0][1]))
    # print("----------------------------")
    col_name = [["Permutation ", "| Final Accuracy "]]
    printtable(col_name, max(len(word)
                             for row in col_name for word in row) + 5)
    for i, element in product(range(len(finals)), finals):
        values = []
        values.append(str(perms[i]))
        values.append("| " + str(element[2]))
        datax = []
        datax.append(values)
        printtable(datax, max(len(word)
                              for row in col_name for word in row) + 5)
        del values, datax
        if finals[0][1] < element[2]:
            accs.append(list([str(perms[i]),
                              "| " + str(element[2])]))
    print("")
    logger("->", " Best Accuracies")
    # print("----------------------------")
    col_name = [["Permutation ", "| Final Accuracy "]]
    printtable(col_name, max(len(word)
                             for row in col_name for word in row) + 5)
    printtable(accs, col_width)

    if inplace:
        data.to_csv(dataset)


def dimensionality_RF(instruction, dataset, target="", y="", n_features=10):
    '''
    function to reduce dimensionality in dataset via random forest method
    :param instruction: command sent to client instance in written query.
    :param dataset: data instantiated in client instance passed to the algorithm
    :param target: column name of response variable/feature
    :param y: dictionary of train/test data values associated with response variable/feature
    :param n_features: maximum number of features to choose to analyze/select
    '''
    
    global counter

    dataReader = DataReader("./data/" + get_last_file()[0])

    if target == "":
        data = dataReader.data_generator()
        data.fillna(0, inplace=True)
        remove = get_similar_column(get_value_instruction(instruction), data)
        data = structured_preprocesser(data)

        y = data[remove]
        del data[remove]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

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

        accuracy_scores.append(accuracy_score(vr.predict(X_temp_test), y_test))

    the_index = accuracy_scores.index(max(accuracy_scores))

    return datas[the_index], accuracy_scores[0], max(
        accuracy_scores), list(columns[the_index])


def dimensionality_PCA(instruction, dataset, ca_threshold=None):
     '''
     function to reduce dimensionality in dataset via principal component analysis method
     :param instruction: command sent to client instance in written query.
     :param dataset: data instantiated in client instance passed to the algorithm
     :param ca_threshold: percentage of dataset to be preprocessed using morphological component analysis
     '''
        
    global counter

    pca = PCA(0.92)

    data, y, target, full_pipeline = initial_preprocesser(
        dataset, instruction, ca_threshold=ca_threshold, preprocess=True)

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
    '''
    function to reduce dimensionality in dataset via independent component analysis
    :param instruction: command sent to client instance in written query.
    :param dataset: data instantiated in client instance passed to the algorithm
    :param target: column name of response variable/feature
    :param y: dictionary of train/test data values associated with response variable/feature
    '''
    global counter

    dataReader = DataReader("./data/" + get_last_file()[0])

    if target == "":
        data = dataReader.data_generator()
        data.fillna(0, inplace=True)
        remove = get_similar_column(get_value_instruction(instruction), data)

        y = data[remove]
        del data[remove]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

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
    acc = []
    acc.append(accuracy_score(
        clf_mod.predict(X_test_mod), y_test_mod))
    for i, j in product(range(3, 10), ["entropy", "gini"]):
        model = tree.DecisionTreeClassifier(criterion=j, max_depth=i)
        model = model.fit(X_train, y_train)
        acc.append(accuracy_score(model.predict(X_test), y_test))
    del i, j
    data_modified = pd.DataFrame(data_modified)
    data_modified[target] = np.r_[y_train, y_test]
    # data_modified.to_csv("./data/housingPCA.csv")

    return data_modified, accuracy_score(
        clf.predict(X_test), y_test), max(acc), (len(
            dataset.columns) - len(data_modified.columns))


def get_last_file():
    '''
    function to retrieve most recently created file generated from
    reduced dimensionality technique
    '''
    
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
    '''
    function to reduce dimensionality in dataset via kernal principal component analysis
    :param instruction: command sent to client instance in written query.
    :param dataset: data instantiated in client instance passed to the algorithm
    :param target: column name of response variable/feature
    :param y: dictionary of train/test data values associated with response variable/feature
    '''
    
    global counter

    dataReader = DataReader("./data/" + get_last_file()[0])

    if target == "":
        data = dataReader.data_generator()
        data.fillna(0, inplace=True)
        remove = get_similar_column(get_value_instruction(instruction), data)

        y = data[remove]
        del data[remove]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

    kpca = KernelPCA(n_components=len(dataset.columns), kernel="rbf")
    data_modified = kpca.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, y, test_size=0.2, random_state=49)
    X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(
        data_modified, y, test_size=0.2, random_state=49)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    clf_mod = tree.DecisionTreeClassifier()
    clf_mod.fit(X_train_mod, y_train_mod)
    acc = []
    acc.append(accuracy_score(
        clf_mod.predict(X_test_mod), y_test_mod))
    for i, j in product(range(3, 10), ["entropy", "gini"]):
        model = tree.DecisionTreeClassifier(criterion=j, max_depth=i)
        model = model.fit(X_train_mod, y_train_mod)
        acc.append(accuracy_score(model.predict(X_test_mod), y_test))
    del i, j
    data_modified = pd.DataFrame(data_modified)
    data_modified[target] = np.r_[y_train, y_test]
    # data_modified.to_csv("./data/housingPCA.csv")

    return data_modified, accuracy_score(
        clf.predict(X_test), y_test), max(acc), (len(
            dataset.columns) - len(data_modified.columns))

    def booster(dataset, obj):
        #obj=["reg:linear","multi:softmax "]

        X_train, X_test, y_train, y_test = train_test_split(
            dataset, y, test_size=0.2, random_state=49)
        clf = XGBClassifier(
            objective=obj,
            learning_rate=0.1,
            silent=1,
            alpha=10)
        clf.fit(X_train, y_train)
        return accuracy_score(clf.predict(X_test_mod), y_test_mod)
        # importance graph
        #plt.rcParams['figure.figsize'] = [5, 5]
        # plt.show()


#dimensionalityPCA("Predict median house value", "./data/housing.csv")

#dimensionalityReduc("Predict ocean_proximity", "./data/housing.csv")

# data = pd.read_csv("./data/housing.csv")
#
# print(dimensionality_PCA("Model ocean proximity", data))
