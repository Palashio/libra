import sys

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def plot_loss(total_loss_train, total_loss_test):
    img = plt.figure()
    plt.plot(total_loss_train)
    plt.plot(total_loss_test)
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    return img
