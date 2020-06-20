from data_reader import DataReader
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband
from tensorflow.keras import layers
from sklearn import preprocessing
import kerastuner as kt
from tensorflow import keras
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense, Input
import keras
import tensorflow as tf
from data_preprocesser import structured_preprocesser, clustering_preprocessor
import sys

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data_generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')


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
                    default=1e-3
                )
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


def tuneReg(
        data,
        target,
        max_layers=10,
        min_layers=2,
        min_dense=32,
        max_dense=512,
        executions_per_trial=3,
        max_trials=3):
    print("entered1")
    # function build model using hyperparameter

    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', min_layers, max_layers)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=min_dense,
                                                max_value=max_dense,
                                                step=32),
                                   activation='relu'))
        model.add(layers.Dense(1, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='mean_squared_error')
        return model

    # random search for the model
    #tuner = RandomSearch(
    #    build_model,
    #    objective='loss',
    #    max_trials=max_trials,
    #    executions_per_trial=executions_per_trial)
    tuner = Hyperband(
        hypermodel,
        max_epochs=max_trials,
        objective='loss',
        seed=42,
        executions_per_trial=executions_per_trial,
        directory='hyperband'
    )

    # tuner.search_space_summary()
    # del data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=49)

    # searches the tuner space defined by hyperparameters (hp) and returns the
    # best model
    tuner.search(X_train, y_train,
                 epochs=5,
                 validation_data=(X_test, y_test),
                 callbacks=[tf.keras.callbacks.TensorBoard('my_dir')])

    models = tuner.get_best_models(num_models=1)
    return models[0]


def tuneClass(
        X,
        y,
        num_classes,
        max_layers=10,
        min_layers=2,
        min_dense=32,
        max_dense=512,
        executions_per_trial=3,
        max_trials=3,
        activation='relu',
        loss='categorical_crossentropy',
        metrics='accuracy'):
    # function build model using hyperparameter
    le = preprocessing.LabelEncoder()
    y = tf.keras.utils.to_categorical(
        le.fit_transform(y), num_classes=num_classes)

    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', min_layers, max_layers)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=min_dense,
                                                max_value=max_dense,
                                                step=32),
                                   activation=activation))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss=loss,
            metrics=[metrics])
        return model

    # tuners, establish the object to look through the tuner search space
    #tuner = RandomSearch(
    #    build_model,
    #    objective='loss',
    #    max_trials=max_trials,
    #    executions_per_trial=executions_per_trial,
    #    directory='models',
    #    project_name='class_tuned')
    
    tuner = Hyperband(
        hypermodel,
        max_epochs=10,
        objective='val_accuracy',
        seed=42,
        executions_per_trial=3,
        directory='hyperband'
    )

    # tuner.search_space_summary()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=49)

    # searches the tuner space defined by hyperparameters (hp) and returns the
    # best model
    tuner.search(X_train, y_train,
                 epochs=5,
                 validation_data=(X_test, y_test))
    models = tuner.get_best_models(num_models=1)
    return models[0]


def tuneCNN(X, y, num_classes):

    # creates hypermodel object based on the num_classes and the input shape
    hypermodel = CNNHyperModel(input_shape=(
        224, 224, 3), num_classes=num_classes)

    # tuners, establish the object to look through the tuner search space
    #tuner = RandomSearch(
    #    hypermodel,
    #    objective='val_accuracy',
    #    seed=42,
    #    max_trials=3,
    #    executions_per_trial=3,
    #    directory='random_search',
    #)
    tuner = Hyperband(
    hypermodel,
    max_epochs=10,
    objective='val_accuracy',
    seed=42,
    executions_per_trial=3,
    directory='hyperband'
    )
    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(X), np.asarray(y), test_size=0.33, random_state=42)

    # searches the tuner space defined by hyperparameters (hp) and returns the
    # best model
    tuner.search(X_train, y_train,
                 validation_data=(X_test, y_test),
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

    # returns the best model
    return tuner.get_best_models(1)[0]
"""
import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, GlobalAveragePooling2D, concatenate
import numpy as np
import sys
sys.path.append('utils/')

import keras
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# use non standard flow_from_directory
from image_preprocessing_ver2 import ImageDataGenerator
# it outputs y_batch that contains onehot targets and logits
# logits came from xception
from keras.layers import Lambda
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras import backend as K

import matplotlib.pyplot as plt


def networkmodule(module_name, x, compress, expand, weight_decay=None, trainable=False):
    #weight_decay=1e-4
    if trainable and weight_decay is not None:
        kernel_regularizer = keras.regularizers.l2(weight_decay) 
    else:
        kernel_regularizer = None
    
    x = Convolution2D(
        compress, (1, 1), 
        name=module_name + '/' + 'compress',
        trainable=trainable, 
        kernel_regularizer=kernel_regularizer
    )(x)
    x = Activation('relu')(x)

    a = Convolution2D(
        expand, (1, 1),
        name=module_name + '/' + 'expand1x1',
        trainable=trainable, 
        kernel_regularizer=kernel_regularizer
    )(x)
    a = Activation('relu')(a)

    b = Convolution2D(
        expand, (3, 3), padding='same',
        name=module_name + '/' + 'expand3x3',
        trainable=trainable, 
        kernel_regularizer=kernel_regularizer
    )(x)
    b = Activation('relu')(b)

    return concatenate([a, b])


def SqueezeNet(weight_decay, image_size=64):

    image = Input(shape=(image_size, image_size, 3))

    x = Convolution2D(
        64, (3, 3), strides=(2, 2), name='conv_1', 
        trainable=False
    )(image) # 111, 111, 64
    
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) 

    x = networkmodule('network_2', x, compress=16, expand=64)
    x = networkmodule('network_3', x, compress=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = networkmodule('network_4', x, compress=32, expand=128)
    x = networkmodule('network_5', x, compress=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = networkmodule('network_6', x, compress=48, expand=192)
    x = networkmodule('network_7', x, compress=48, expand=192)
    x = networkmodule('network_8', x, compress=64, expand=256)
    x = networkmodule('network_9', x, compress=64, expand=256)
    
    x = Dropout(0.5)(x)
    x = Convolution2D(
        256, (1, 1), name='conv_10',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    
    x = Activation('relu')(x)
    logits = GlobalAveragePooling2D()(x)
    probabilities = Activation('softmax')(logits)
    
    model = Model(image, probabilities)
    model.load_weights('squeezenet_weights.hdf5', by_name=True)
    
    return model


model1=SqueezeNet(None)
# Compiling the CNN
model=Sequential()
model.add(model1)
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

val_generator = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


###############################################################################

temperature = 5.0
model = SqueezeNet(weight_decay=1e-4, image_size=299)

# remove softmax
model.layers.pop()

# usual probabilities
logits = model.layers[-1].output
probabilities = Activation('softmax')(logits)

# softed probabilities
logits_T = Lambda(lambda x: x/temperature)(logits)
probabilities_T = Activation('softmax')(logits_T)

output = concatenate([probabilities, probabilities_T])
model = Model(model.input, output)
# now model outputs 512 dimensional vectors

def knowledge_distillation_loss(y_true, y_pred, lambda_const):    
    
    # split in 
    #    onehot hard true targets
    #    logits from xception
    y_true, logits = y_true[:, :256], y_true[:, 256:]
    
    # convert logits to soft targets
    y_soft = K.softmax(logits/temperature)
    
    # split in 
    #    usual output probabilities
    #    probabilities made softer with temperature
    y_pred, y_pred_soft = y_pred[:, :256], y_pred[:, 256:]    
    
    return lambda_const*logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)

def accuracy(y_true, y_pred):
    y_true = y_true[:, :256]
    y_pred = y_pred[:, :256]
    return categorical_accuracy(y_true, y_pred)


def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:, :256]
    y_pred = y_pred[:, :256]
    return top_k_categorical_accuracy(y_true, y_pred)

def categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, :256]
    y_pred = y_pred[:, :256]
    return logloss(y_true, y_pred)


# logloss with only soft probabilities and targets
def soft_logloss(y_true, y_pred):     
    logits = y_true[:, 256:]
    y_soft = K.softmax(logits/temperature)
    y_pred_soft = y_pred[:, 256:]    
    return logloss(y_soft, y_pred_soft)

lambda_const = 0.2

model.compile(
    optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True), 
    loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const), 
    metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss]
)
model.fit_generator(
    train_generator, 
    steps_per_epoch=400, epochs=30, verbose=1,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.01),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, epsilon=0.007)
    ],
    validation_data=val_generator, validation_steps=80, workers=4
)

val_generator_no_shuffle = data_generator.flow_from_directory(
    data_dir + 'val', val_logits,
    target_size=(299, 299),
    batch_size=64, shuffle=False
)
print(model.evaluate_generator(val_generator_no_shuffle, 80))
"""