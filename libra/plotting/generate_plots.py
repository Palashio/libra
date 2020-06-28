import sys

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from numpy import interp
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

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


def plot_mc_roc(y_test, y_score, interpreter=None):
    lw = 2
    n_classes = len(np.unique(y_test))
    classes = pd.unique(y_test)
    y_test = label_binarize(y_test, classes=classes)
    y_score = label_binarize(y_score, classes=classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    for i in range(n_classes):
        if isinstance(interpreter, dict):
            inverted_interpreter = dict(map(reversed, interpreter.items()))
            plt.plot(fpr[i], tpr[i], lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(inverted_interpreter[i], roc_auc[i]))
        else:
            plt.plot(
                fpr[i],
                tpr[i],
                lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(
                    interpreter.inverse_transform(
                        [i])[0],
                    roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()
