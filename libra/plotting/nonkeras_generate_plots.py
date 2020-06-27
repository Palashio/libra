import sys

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def plot_loss(total_loss_train,total_loss_test):
    img = plt.figure()
    plt.plot(total_loss_train)
    plt.plot(total_loss_test)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    return img

def plot_accuracy(accuracy_train, accuracy_test):
    img = plt.figure()
    plt.plot(accuracy_train)
    plt.plot(accuracy_test)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['test','val'], loc='upper left')
    return img