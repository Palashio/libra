import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from numpy import interp
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, recall_score, precision_score, f1_score, ConfusionMatrixDisplay
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


currLog = ""
counter = 0
number = 0
# current_dir=os.getcw()

# allows for all columns to be displayed when printing()
pd.options.display.width = None


# clears the log when new process is started up


def clearLog():
    global currLog
    global counter

    currLog = ""
    counter = 0


# logging function that creates hierarchial display of the processes of
# different functions. Copied into different python files to maintain
# global variable parallels


def logger(instruction, found=""):
    global currLog
    global counter
    if counter == 0:
        currLog += (" " * 2 * counter) + str(instruction) + str(found)
    elif instruction == "->":
        counter = counter - 1
        currLog += (" " * 2 * counter) + str(instruction) + str(found)
    else:
        currLog += (" " * 2 * counter) + "|- " + str(instruction) + str(found)
        if instruction == "done...":
            currLog += "\n" + "\n"

    counter += 1
    print(currLog)
    currLog = ""


# generates all of the plots in clustering


def generate_clustering_plots(kmeans, dataPandas, dataset):
    plots = []
    plot_names = []
    # for all of the columns that're present in clustering it plots the
    # columns with each other based on the cluster they're in
    for x in range(len(dataPandas.columns) - 1):
        for y in range(len(dataPandas.columns) - 1):
            if dataPandas.columns[x] != dataPandas.columns[y]:
                img = plt.figure()
                plt.scatter(dataset[:, x], dataset[:, y],
                            c=kmeans.labels_, cmap='rainbow')
                plt.xlabel(str(dataPandas.columns[x]))
                plt.ylabel(str(dataPandas.columns[y]))
                plots.append(img)
                plot_names.append(
                    dataPandas.columns[x] +
                    "_vs_" +
                    dataPandas.columns[y])
                plt.close(img)
    return plots, plot_names

# generates all of the plots for regression


def generate_regression_plots(history, data, label):
    plots = []
    plot_names = []
    # generating plots for loss
    plots.append(plot_loss(history))
    plot_names.append('lossvsval_los')

    return plots, plot_names


def generate_classification_plots(history, data, label, model, X_test, y_test):
    plots = []
    plot_names = []

    # generating plots for accuracy
    plots.append(plot_acc(history))
    plot_names.append('accuracyvsval_accuracy')
    # generating plots for loss
    plots.append(plot_loss(history))
    plot_names.append('lossvsval_los')

    # dynamic way to return all possible plots in case it expands together
    return_plots = {}
    for x in range(len(plots)):
        return_plots[str(plot_names[x])] = plots[x]

    return return_plots

# function to return both val and accuracy plots on one pane


def generate_classification_together(history, data, model, X_test, y_test):
    plots = []
    plot_names = []

    arrEpochs = []
    # stores all of the history information
    for x in range(len(history.history['loss'])):
        arrEpochs.append(x + 1)

    # the first loss plot on the top
    plt.subplot(2, 1, 1)
    plot_loss(history)
    # the second accuracy plot on the bottom
    plt.subplot(2, 1, 2)
    plot_acc(history)
    plt.show()


def plot_loss(history):
    img = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return img


def plot_corr(data, col=[]):
    # Here, col is a string list which indicates the columns between
    # which the correlation is required. if left empty it shows the
    # correlation heatmap for all the variables.
    img = plt.figure()
    if col:
        data = data.loc[:, data.columns.intersection(['a', 'b'])]
    corr = data.corr()
    sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        square=True)
    return img


def plot_acc(history):
    img = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return img
