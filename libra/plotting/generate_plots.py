import seaborn as sns
import warnings
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from numpy import interp
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, recall_score, precision_score, f1_score, \
    ConfusionMatrixDisplay
import numpy as np
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
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


def logger(instruction, found=""):
    '''
    logging function that creates hierarchial display of the processes of
    different functions. Copied into different python files to maintain
    global variables.

    :param instruction: what you want to be displayed
    :param found: if you want to display something found like target column
    '''
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


def generate_clustering_plots(
        kmeans,
        dataPandas,
        dataset,
        scatters,
        inertia_sor,
        base_clusters):
    '''
    plotting function that generates all plots for the clustering algorithm

    :param base_clusters: is the number of clusters testing began at
    :param inertia_sor: the array that has all the inertia values
    :param scatters: which scatter plots you want to generate of columns compared to each other
    :param dataset: is the dataset that you want to generate plots for
    :param dataPandas: a pandas version of that same set
    :param kmeans: is the actual kmeans algorithm you're generating for
    :return an array with all of the plots stored as figures inside of it
    '''

    plots = []
    plot_names = []
    # for all of the columns that're present in clustering it plots the
    # columns with each other based on the cluster they're in
    for x in range(len(dataPandas.columns) - 1):
        for y in range(len(dataPandas.columns) - 1):
            if dataPandas.columns[x].replace(
                    " ", "_") + "_vs_" + dataPandas.columns[y].replace(" ", "_") in scatters:
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
    return plots, plot_names, elbow_cluster_graph(inertia_sor, base_clusters)


# generates all of the plots for regression
def elbow_cluster_graph(inertia_stor, base_clusters):
    '''
    plotting function that generates the elbow graph

    :param base_clusters: number of clusters you begin testing at
    :param inertia_stor: the array of inertia values
    :return the elbow graph
    '''
    ranged = []
    for i in range(base_clusters, len(inertia_stor) + base_clusters):
        ranged.append(i + 1)

    img = plt.figure()
    plt.plot(ranged, inertia_stor, marker='o')
    # plt.plot(inertia_stor)
    plt.title('Elbow Graph for Clustering')
    plt.ylabel('SSE/Inertia')
    plt.xlabel('Number of Clusters')
    return img


def generate_regression_plots(history, data, label):
    '''
    plotting function that generates regression plots

    :param history: the keras history object
    :return the names and actual regression plots
    '''
    plots = []
    plot_names = []
    # generating plots for loss
    plots.append(plot_loss(history))
    plot_names.append('lossvsval_los')

    return plots, plot_names


def generate_classification_plots(history, data, label, model, X_test, y_test):
    '''
    plotting function that generates classification plots

    :param history: the keras history object
    :return the names and actual classification plots
    '''
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
    '''
    unused plotting function that generates multiple plots on one pane when defaulted

    :param history: is the history object of keras
    :param many params: information that is given to the plots for specifications
    :return plots on the same pane
    '''
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
    '''
    plotting function that generates loss plots

    :param history: is the history object from keras
    :return a plot of validation vs training loss
    '''
    img = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return img


def plot_corr(data, col=[]):
    '''
    correlation matrix for plot roc

    :param data: is the data to find correlations for
    :param col: is the specific column you want to find correlations to
    :return correlation matrix
    '''

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
    '''
    plotting function that generates accuracy plots

    :param history: is the history object from keras
    :return a plot of validation vs training accuracy
    '''
    img = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return img


def plot_mc_roc(y_test, y_score, interpreter=None):
    '''
    plotting function that generates roc curves for data given to it.

    :param y_test: is the testing data used
    :param y_score: is the score when the testing data was called
    :param interpreter: is what was used to preprocess
    :return a roc plot
    '''
    lw = 2
    n_classes = len(np.unique(y_test))
    classes = pd.unique(y_test)
    if n_classes != 2:
        y_test = label_binarize(y_test, classes=classes)
        y_score = label_binarize(y_score, classes=classes)
    else:
        n_classes = 1
        y_test = y_test.reshape(-1, 1)
        y_score = y_score.reshape(-1, 1)

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
    img = plt.figure()
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
                        [[i]])[0],
                    roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    return img


# Analysis of model
def analyze(client, model=None, save=True, save_model=False):
    '''
    the body of the analyze function in queries.py. Used to generate ROC, confusion matrix etc.
    :param model: is the actual model that you want to analyze for and against
    :param client: is the whole client object :)
    :param save: whether to save into client dictionary
    :param save_model: whether to save as .png
    '''

    if model is None:
        model = client.latest_model

    if not model in client.models:
        raise NameError('Model name is not valid or has not been created yet.')

    plt.clf()
    logger("Analyzing {} for further understanding".format(model))

    modeldict = client.models[model]

    if 'test_data' in modeldict:
        logger("Making predictions for test data")
        data = modeldict['test_data']['X']
        real = modeldict['test_data']['y']
        preds = modeldict['model'].predict(data)
        if model == 'Text Classification':
            logger("Identifying textual relationships and numerical correspondence")
            preds = np.argmax(preds, axis=-1)
        if model == 'classification_ANN':  # formats labels column
            enc = sklearn.preprocessing.LabelEncoder()
            real = modeldict['interpreter'].inverse_transform(
                real).reshape(1, -1)[0]
            preds = modeldict['interpreter'].inverse_transform(
                preds).reshape(1, -1)[0]
            real = enc.fit_transform(real)
            preds = enc.transform(preds)
            logger("Transforming dataset for plotting module")

    if model == 'k_means_clustering':
        logger("Gathering metrics for display: ")
        inertia = modeldict['model'].inertia_
        centers = modeldict['model'].cluster_centers_
        logger(" ", ("Optimal number of clusters: {}".format(str(len(centers)))))
        logger(
            "->",
            ("Sum of squared distance or inertia: {}".format(
                str(inertia))))
        if save:
            modeldict['n_centers'] = len(centers)
            modeldict['centroids'] = centers
            modeldict['inertia'] = inertia
    elif model == 'regression_ANN':
        logger("Gathering metrics for display: ")
        MSE = sklearn.metrics.mean_squared_error(real, preds)
        MAE = sklearn.metrics.mean_absolute_error(real, preds)
        logger(" ", ("MSE on test set: {}".format(str(MSE))))
        logger("->", ("MAE on test set: {}".format(str(MAE))))
        if save:
            modeldict['MSE'] = MSE
            modeldict['MAE'] = MAE
    # classification models
    elif model in ['svm', 'nearest_neighbor', 'decision_tree', 'classification_ANN', 'Text Classification']:
        logger("->", "Plotting ROC curves and creating confusion matrix...")
        if model in ['svm', 'nearest_neighbor',
                     'decision_tree', 'Text Classification']:
            label_source = modeldict['interpreter']
            labels = list(modeldict['interpreter'].keys())
        else:
            label_source = enc
            labels = enc.classes_

        # create roc plots
        roc = plot_mc_roc(real, preds, label_source)

        # create confusion matrices
        cm = confusion_matrix(real, preds)
        cm = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels).plot()
        cm = cm.figure_

        logger('Investigating potential issues with calculations')
        logger("Gathering metrics for display: ")
        # get accuracy from modeldict
        if model in ['svm', 'nearest_neighbor',
                     'decision_tree']:
            accuracy = modeldict['accuracy']['accuracy_score']
        else:
            accuracy = modeldict['accuracy']['validation_accuracy']
        recall = recall_score(real, preds, average='micro')
        precision = precision_score(real, preds, average='micro')
        f1 = f1_score(real, preds, average='micro')

        logger("->",
               ("Accuracy on test set: {}".format(str((accuracy[-1] if isinstance(accuracy, list) else accuracy)))))
        logger("->", ("Recall on test set: {}".format(str(recall))))
        logger("->", ("Precision on test set: {}".format(str(precision))))
        logger("->", ("F1 Score on test set: {}".format(str(f1))))
        if save:
            if 'plots' not in modeldict:
                modeldict['plots'] = {}
            modeldict['plots']['roc_curve'] = roc
            modeldict['plots']['confusion_matrix'] = cm

            if 'scores' not in modeldict:
                modeldict['scores'] = {}
            modeldict['scores']['recall_score'] = recall
            modeldict['scores']['precision_score'] = precision
            modeldict['scores']['f1_score'] = f1

    else:
        print("further analysis is not supported for {}".format(model))

    if 'plots' in modeldict and model != 'k_means_clustering':
        client.plots(model=model, save=save_model)

    clearLog()
