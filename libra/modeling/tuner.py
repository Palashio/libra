import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from kerastuner.applications import HyperResNet
from tensorflow.keras.layers import (Conv2D,
                                     MaxPooling2D,
                                     Dense,
                                     Flatten,
                                     Dropout)
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband
from sklearn import preprocessing
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# creates hypermodel class for CNN tuning

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    # model builder: creates different layers with a selection of parameters
    # that during runtime it selects to use
    def build(self, hp):
        model = keras.Sequential()
        model.add(
            Conv2D(
                filters=16,
                kernel_size=3,
                activation='relu',
                input_shape=self.input_shape
            )
        )
        model.add(
            Conv2D(
                filters=16,
                activation='relu',
                kernel_size=3
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(
            Conv2D(
                filters=32,
                kernel_size=3,
                activation='relu'
            )
        )
        model.add(
            Conv2D(
                filters=16,
                activation='relu',
                kernel_size=3
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_2',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(Flatten())
        model.add(
            Dense(
                units=hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )
        model.add(
            Dropout(
                rate=hp.Float(
                    'dropout_3',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )
            )
        )
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3)),
            loss=(
                'binary_crossentropy' if self.num_classes == 2 else 'categorical_crossentropy'),
            metrics=['accuracy'])
        return model


def tuneReg(
        data,
        target,
        max_layers=10,
        min_layers=2,
        min_dense=32,
        max_dense=512,
        executions_per_trial=1,
        max_trials=5,
        epochs=10,
        activation='relu',
        directory='my_dir',
        step=32,
        verbose=0,
        test_size=0.2
):
    # function build model using hyperparameter
    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', min_layers, max_layers)):
            model.add(Dense(units=hp.Int('units_' + str(i),
                                         min_value=min_dense,
                                         max_value=max_dense,
                                         step=step),
                            activation=activation))
            model.add(Dropout(rate=hp.Float(
                              'dropout_3',
                              min_value=0.0,
                              max_value=0.5,
                              default=0.20,
                              step=0.05)))
        model.add(Dense(1, activation='linear'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                                       hp.Float('learning_rate',
                                                min_value=1e-5,
                                                max_value=1e-2,
                                                sampling='LOG',
                                                default=1e-3)),
            loss='mse',
            metrics=['accuracy'])
        return model

    # random search for the model
    tuner = RandomSearch(
        build_model,
        objective='loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory)
    # tuner.search_space_summary()
    # del data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=49)

    # searches the tuner space defined by hyperparameters (hp) and returns the
    # best model

    tuner.search(X_train, y_train,
                 epochs=epochs,
                 validation_data=(X_test, y_test),
                 callbacks=[tf.keras.callbacks.TensorBoard('my_dir')])

    models = tuner.get_best_models(num_models=1)[0]
    hyp = tuner.get_best_hyperparameters(num_trials=1)[0]
    history = tuner_hist(
        data,
        target,
        tuner,
        hyp,
        epochs=epochs,
        verbose=verbose,
        test_size=test_size)
    """
    Return:
        models[0] : best model obtained after tuning
        best_hps : best Hyperprameters obtained after tuning, stored as map
        history : history of the data executed from the given model
    """
    return models, hyp, history, X_test, y_test


def tuneClass(
        X,
        y,
        num_classes,
        max_layers=10,
        min_layers=2,
        min_dense=32,
        max_dense=512,
        executions_per_trial=1,
        max_trials=3,
        activation='relu',
        directory='my_dir',
        loss='categorical_crossentropy',
        metrics='accuracy',
        epochs=10,
        step=32,
        verbose=0,
        test_size=0.2):
    # function build model using hyperparameter
    le = preprocessing.LabelEncoder()
    y = tf.keras.utils.to_categorical(
        le.fit_transform(y), num_classes=num_classes)

    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', min_layers, max_layers)):
            model.add(Dense(units=hp.Int('units_' + str(i),
                                         min_value=min_dense,
                                         max_value=max_dense,
                                         step=step),
                            activation=activation))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                                       hp.Float('learning_rate',
                                                min_value=1e-5,
                                                max_value=1e-2,
                                                sampling='LOG',
                                                default=1e-3)),
            loss=loss,
            metrics=[metrics])
        return model
    
    # tuners, establish the object to look through the tuner search space
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory,
        project_name='class_tuned')

    # tuner.search_space_summary()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=49)

    # searches the tuner space defined by hyperparameters (hp) and returns the
    # best model
    tuner.search(X_train, y_train,
                 epochs=epochs,
                 validation_data=(X_test, y_test))
    models = tuner.get_best_models(num_models=1)[0]
    hyp = tuner.get_best_hyperparameters(num_trials=1)[0]
    #hyp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values
    #best_hps = np.stack(hyp).astype(None)
    history = tuner_hist(
        X,
        y,
        tuner,
        hyp,
        epochs=epochs,
        verbose=verbose,
        test_size=test_size)

    """
    Return:
        models[0] : best model obtained after tuning
        best_hps : best Hyperprameters obtained after tuning, stored as array
        history : history of the data executed from the given model
    """
    return models, hyp, history, X_test, y_test


def tuneCNN(
        X_train,
        X_test,
        height,
        width,
        num_classes,
        patience=1,
        executions_per_trial=1,
        seed=42,
        max_trials=3,
        objective='val_accuracy',
        directory='my_dir',
        epochs=10,
        verbose=0,
        test_size=0.2):
    # creates hypermodel object based on the num_classes and the input shape
    hypermodel = CNNHyperModel(input_shape=(
        height, width, 3), num_classes=num_classes)

    # # tuners, establish the object to look through the tuner search space
    tuner = RandomSearch(
        hypermodel,
        objective=objective,
        seed=seed,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory,
    )


    # searches the tuner space defined by hyperparameters (hp) and returns the
    # best model

    tuner.search(X_train,
                 validation_data=X_test,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience)],
                 epochs=epochs,
                 verbose=verbose)

    # best hyperparamters
    hyp = tuner.get_best_hyperparameters(num_trials=1)[0]
    #hyp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values
    #best_hps = np.stack(hyp).astype(None)
    history = tuner_hist(
        X_train,
        X_test,
        tuner,
        hyp,
        img=1,
        epochs=epochs,
        verbose=verbose,
        test_size=test_size)

    """
    Return:
        models[0] : best model obtained after tuning
        best_hps : best Hyperprameters obtained after tuning, stored as array
        history : history of the data executed from the given model
    """
    return tuner.get_best_models(1)[0], hyp, history


def tuneHyperband(X,
                  y,
                  max_trials=3):
    """
    Perform Hyperband Tuning to search for the best model and Hyperparameters
    Arguments:
        X: Input dataset
        y: Label or output dataset
        max_trials: Trials required to perform tuning
    """
    hypermodel = HyperResNet(input_shape=(128, 128, 3), num_classes=10)
    tuner = Hyperband(
        hypermodel,
        max_epochs=max_trials,
        objective='val_accuracy',
        seed=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=49)

    # searches the tuner space defined by hyperparameters (hp) and returns the
    # best model
    tuner.search(X_train, y_train,
                 epochs=5,
                 validation_data=(X_test, y_test))
    hyp = tuner.get_best_hyperparameters(num_trials=1)[0]
    #hyp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values
    #best_hps = np.stack(hyp).astype(None)

    history = tuner_hist(X, y, tuner, hyp)
    """
    Return:
        models[0] : best model obtained after tuning
        best_hps : best Hyperprameters obtained after tuning, stored as array
        history : history of the data executed from the given model
    """
    return tuner.get_best_models(1)[0], hyp, history


def tuner_hist(
        X,
        y,
        tuner,
        best_hps,
        img=0,
        epochs=5,
        test_size=0.2,
        verbose=0):
    model = tuner.hypermodel.build(best_hps)

    if img == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=49)
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            verbose=verbose)
    else:
        history = model.fit_generator(X,
                                      epochs=epochs,
                                      validation_data=y,
                                      verbose=verbose)

    return history
