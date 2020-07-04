import sys

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


def plot_loss(total_loss_train, total_loss_test):
    img = plt.figure()
    plt.plot(total_loss_train)
    plt.plot(total_loss_test)
    plt.ylim(0)
    plt.xticks((np.arange(0, len(total_loss_train), step=1)))
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    return img
