import sys
import os

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data_generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')

from image_preprocesser import setwise_preprocessing, pathwise_preprocessing, classwise_preprocessing
from data_reader import DataReader
from keras.models import Sequential
from keras.layers import (Dense, Conv2D, Flatten, Input, MaxPooling2D, )
import pandas as pd
from supplementaries import save, generate_id
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from generate_plots import (generate_clustering_plots,
                           generate_regression_plots,
                           generate_classification_plots)
from data_preprocesser import structured_preprocesser, initial_preprocesser
from prediction_model_creation import get_keras_model_reg, get_keras_text_class
from prediction_model_creation import get_keras_model_class
from sklearn.preprocessing import StandardScaler
import numpy as np
#from prediction_queries import logger, clearLog


currLog = ""
counter = 0
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


def logger(instruction, found="", slash=''):
    global currLog
    global counter
    if counter == 0:
        currLog += (" " * 2 * counter) + str(instruction) + str(found)
    elif instruction == "->":
        counter = counter - 1
        if slash == '|':
            currLog += (" " * 2 * counter) + slash + str(found)
        else:
            currLog += (" " * 2 * counter) + str(instruction) + str(found)
    else:
        currLog += (" " * 2 * counter) + "|" + "\n"
        currLog += (" " * 2 * counter) + "|- " + str(instruction) + str(found)
        if instruction == "done...":
            currLog += "\n" + "\n"

    counter += 1
    if instruction == "->":
        print(currLog, end="")
    else:
        print(currLog)
    currLog = ""

def regression_ann(
            instruction,
            dataset = None,
            drop=None,
            preprocess=True,
            test_size=0.2,
            random_state=49,
            epochs=50,
            generate_plots=True,
            callback_mode='min',
            maximizer="val_loss",
            save_model=True,
            save_path=os.getcwd()):

        global currLog
        logger("reading in dataset...")

        dataReader = DataReader(dataset)
        data = dataReader.data_generator()
        # data = pd.read_csv(self.dataset)

        if drop is not None:
            data.drop(drop, axis=1, inplace=True)

        data, y, target, full_pipeline = initial_preprocesser(data, instruction, preprocess)

        X_train = data['train']
        X_test = data['test']


        # Target scaling
        target_scaler = StandardScaler()

        y_train = target_scaler.fit_transform(np.array(y['train']).reshape(-1, 1))
        y_test = target_scaler.transform(np.array(y['test']).reshape(-1,1))

        logger("establishing callback function...")

        models = []
        losses = []
        model_data = []

        # callback function to store lowest loss value
        es = EarlyStopping(
            monitor=maximizer,
            mode=callback_mode,
            verbose=0,
            patience=5)

        i = 0

        # get the first 3 layer model
        model = get_keras_model_reg(data, i)

        logger("training initial model...")
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(
                X_test,
                y_test),
            callbacks=[es],
            verbose=0)
        models.append(history)
        model_data.append(model)

        logger("->", "Initial number of layers " + str(len(model.layers)))

        logger("->", "Training Loss: " + \
               str(history.history['loss'][len(history.history['val_loss']) - 1]), '|')
        logger("->", "Test Loss: " +
               str(history.history['val_loss'][len(history.history['val_loss']) -
                                               1]), '|')
        print("")

        losses.append(history.history[maximizer]
                      [len(history.history[maximizer]) - 1])

        # keeps running model and fit functions until the validation loss stops
        # decreasing
        logger("testing number of layers...")
        print(currLog)
        while (all(x > y for x, y in zip(losses, losses[1:]))):
            model = get_keras_model_reg(data, i)
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_data=(
                    X_test,
                    y_test), verbose=0)
            model_data.append(model)
            models.append(history)
            logger("->", "Current number of layers: " + str(len(model.layers)))

            logger("->", "Training Loss: " +
                   str(history.history['loss'][len(history.history['val_loss']) -
                                               1]), '|')
            logger("->", "Test Loss: " +
                   str(history.history['val_loss'][len(history.history['val_loss']) -
                                                   1]), '|')
            print("")
            losses.append(history.history[maximizer]
                          [len(history.history[maximizer]) - 1])
            i += 1

        final_model = model_data[losses.index(min(losses))]
        final_hist = models[losses.index(min(losses))]

        logger('->', "Best number of layers found: " +
               str(len(final_model.layers)))

        logger('->', "Training Loss: " + str(final_hist.history['loss']
                                             [len(final_hist.history['val_loss']) - 1]))
        logger('->', "Test Loss: " + str(final_hist.history['val_loss']
                                         [len(final_hist.history['val_loss']) - 1]))

        # calls function to generate plots in plot generation
        if generate_plots:
            init_plots, plot_names = generate_regression_plots(
                models[len(models) - 1], data, y)
            plots = {}
            for x in range(len(plot_names)):
                plots[str(plot_names[x])] = init_plots[x]

        if save_model:
            save(final_model, save_model)
        # stores values in the client object models dictionary field
        return {
            'id': generate_id(),
            'model': final_model,
            "target": target,
            "plots": plots,
            "preprocesser": full_pipeline,
            "interpreter": target_scaler,
            'losses': {
                'training_loss': final_hist.history['loss'],
                'val_loss': final_hist.history['val_loss']}}


def classification_ann(instruction,
            dataset=None,
            preprocess=True,
            callback_mode='min',
            drop=None,
            random_state=49,
            test_size=0.2,
            epochs=5,
            generate_plots=True,
            maximizer="val_loss",
            save_model=True,
            save_path=os.getcwd()):

        dataReader = DataReader(dataset)
        data = dataReader.data_generator()

        if drop is not None:
            data.drop(drop, axis=1, inplace=True)

        data, y, remove, full_pipeline = initial_preprocesser(
            data, instruction, preprocess)

        # Needed to make a custom label encoder due to train test split changes
        # Can still be inverse transformed, just a bit of extra work
        y = pd.concat([y['train'], y['test']], axis=0)

        num_classes = len(np.unique(y))

        X_train = data['train']
        X_test = data['test']

        # ANN needs target one hot encoded for classification
        one_hot_encoder = OneHotEncoder()

        y = pd.DataFrame(one_hot_encoder.fit_transform(np.reshape(y.values, (-1,1))).toarray(),
                         columns=one_hot_encoder.get_feature_names())

        y_train = y.iloc[:len(X_train)]
        y_test = y.iloc[len(X_train):]


        models = []
        losses = []
        accuracies = []
        model_data = []

        # early stopping callback
        es = EarlyStopping(
            monitor=maximizer,
            mode='min',
            verbose=0,
            patience=5)

        i = 0
        model = get_keras_model_class(data, i, num_classes)

        history = model.fit(
            X_train, y_train, epochs=epochs, validation_data=(
                X_test, y_test), callbacks=[es], verbose=0)

        model_data.append(model)
        models.append(history)
        logger("->", "Initial number of layers: " + str(len(model.layers)))

        logger("->", "Training Loss: " +
               str(history.history['loss'][len(history.history['val_loss']) - 1]), '|')
        logger("->", "Test Loss: " +
               str(history.history['val_loss'][len(history.history['val_loss']) -
                                               1]), '|')
        print("")

        losses.append(history.history[maximizer]
                      [len(history.history[maximizer]) - 1])

        # keeps running model and fit functions until the validation loss stops
        # decreasing
        logger("testing number of layers...")
        while (all(x > y for x, y in zip(losses, losses[1:]))):
            model = get_keras_model_class(data, i, num_classes)
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_data=(
                    X_test,
                    y_test),
                callbacks=[es], verbose=0)

            model_data.append(model)
            models.append(history)
            logger("->", "Current number of layers: " + str(len(model.layers)))

            logger("->", "Training Loss: " +
                   str(history.history['loss'][len(history.history['val_loss']) -
                                               1]), '|')
            logger("->", "Test Loss: " +
                   str(history.history['val_loss'][len(history.history['val_loss']) -
                                                   1]), '|')
            print("")

            losses.append(history.history[maximizer]
                          [len(history.history[maximizer]) - 1])
            accuracies.append(history.history['val_accuracy']
                              [len(history.history['val_accuracy']) - 1])
            i += 1

        final_model = model_data[losses.index(min(losses))]
        final_hist = models[losses.index(min(losses))]

        logger('->', "Best number of layers found: " +
               str(len(final_model.layers)))
        logger('->', "Training Accuracy: " + str(final_hist.history['accuracy']
                                                 [len(final_hist.history['val_accuracy']) - 1]))
        logger('->', "Test Accuracy: " + str(final_hist.history['val_accuracy'][
                                                 len(final_hist.history['val_accuracy']) - 1]))

        # genreates appropriate classification plots by feeding all information
        plots = generate_classification_plots(
            models[len(models) - 1], data, y, model, X_test, y_test)

        # if save_model:
        #     save(final_model, save_model)

        # stores the values and plots into the object dictionary
        return {
            'id': generate_id(),
            "model": final_model,
            'num_classes': num_classes,
            "plots": plots,
            "target": remove,
            "preprocesser": full_pipeline,
            "interpreter": one_hot_encoder,
            'losses': {
                'training_loss': final_hist.history['loss'],
                'val_loss': final_hist.history['val_loss']},
            'accuracy': {
                'training_accuracy': final_hist.history['accuracy'],
                'validation_accuracy': final_hist.history['val_accuracy']}}


def convolutional(self,
                read_mode="setwise",
                data_paths=None,
                new_folders=True,
                csv_file=None,
                label_column=None,
                image_column=None,
                training_ratio=0.8):

    logger("Creating CNN generation query")
    # generates the dataset based on instructions using a selenium query on
    # google chrome
    logger("Generating datasets for classes...")

    # if image dataset in form of a data folder
    training_path = "/proc_training_set"
    testing_path = "/proc_testing_set"

    if read_mode=="setwise":
        if data_paths is None:
            data_path = os.getcwd()
        else:
            data_path = data_paths
        # process images
        processInfo = setwise_preprocessing(data_paths, new_folders)
        if not new_folders:
            training_path = "/training_set"
            testing_path = "/testing_set"

    # if image dataset in form of csv
    elif read_mode=="pathwise":
        processInfo = pathwise_preprocessing(csv_file, data_paths, label_column, image_column, training_ratio)
        data_path = os.path.dirname(csv_file)

    # if image dataset in form of one folder containing class folders
    elif read_mode=="classwise":
        processInfo = classwise_preprocessing(data_paths, training_ratio)
        data_path = data_paths

    input_shape = (processInfo["height"], processInfo["width"], 3)
    input_single = (processInfo["height"], processInfo["width"])
    num_classes = processInfo["num_categories"]
    loss_func = ""

    if num_classes > 2:
        loss_func = "categorical_crossentropy"
    elif num_classes == 2:
        loss_func = "binary_crossentropy"

    logger("Creating convolutional neural network dynamically...")
    # Convolutional Neural Network
    model = Sequential()
    model.add(
        Conv2D(
            64,
            kernel_size=3,
            activation="relu",
            input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss=loss_func,
        metrics=['accuracy'])

    train_data = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    X_train = train_data.flow_from_directory(data_path + training_path,
                                                target_size=input_single,
                                                color_mode='rgb',
                                                batch_size=32,
                                                class_mode='categorical')
    test_data = ImageDataGenerator(rescale=1. / 255)
    X_test = test_data.flow_from_directory(data_path + testing_path,
                                            target_size=input_single,
                                            color_mode='rgb',
                                            batch_size=32,
                                            class_mode='categorical')

    # print(X_train)
    history = model.fit_generator(
        generator=X_train,
        steps_per_epoch=X_train.n //
        X_train.batch_size,
        validation_data=X_test,
        validation_steps=X_test.n //
        X_test.batch_size,
        epochs=10)

    # storing values the model dictionary
    return {
        'id': generate_id(),
        "model": model,
        'num_classes': (2 if num_classes == 1 else num_classes),
        'losses': {
            'training_loss': history.history['loss'],
            'val_loss': history.history['val_loss']},
        'accuracy': {
            'training_accuracy': history.history['accuracy'],
            'validation_accuracy': history.history['val_accuracy']}}