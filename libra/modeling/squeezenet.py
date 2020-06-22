import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Lambda
from keras.layers import Activation, Dropout, GlobalAveragePooling2D, concatenate

import keras
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# use non standard flow_from_directory
from image_preprocessing_ver2 import ImageDataGenerator
# it outputs y_batch that contains onehot targets and logits
# logits came from xception
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras import backend

temperature, lambda_const = 5.0, 0.2
num_classes=2

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

# Part 2 - Fitting the CNN to the images
def knowledge_distillation_loss(y_true, y_pred, lambda_const,num_classes):    
    
    # split in 
    #    onehot hard true targets
    #    logits from xception
    y_true, logits = y_true[:, :num_classes], y_true[:, num_classes:]
    
    # convert logits to soft targets
    y_soft = backend.softmax(logits/temperature)
    
    # split in 
    #    usual output probabilities
    #    probabilities made softer with temperature
    y_pred, y_pred_soft = y_pred[:, :num_classes], y_pred[:, num_classes:]    
    
    return lambda_const*logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)

def accuracy(y_true, y_pred):
    y_true = y_true[:, :num_classes]
    y_pred = y_pred[:, :num_classes]
    return categorical_accuracy(y_true, y_pred)


def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:, :num_classes]
    y_pred = y_pred[:, :num_classes]
    return top_k_categorical_accuracy(y_true, y_pred)

def categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, :num_classes]
    y_pred = y_pred[:, :num_classes]
    return logloss(y_true, y_pred)


# logloss with only soft probabilities and targets
def soft_logloss(y_true, y_pred):     
    logits = y_true[:, num_classes:]
    y_soft = backend.softmax(logits/temperature)
    y_pred_soft = y_pred[:, num_classes:]    
    return logloss(y_soft, y_pred_soft)

def get_snet_layer():
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


    model = SqueezeNet(weight_decay=1e-4, image_size=64)

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
    """
    model.compile(
        optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True), 
        loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const), 
        metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss]
    )
    model.fit_generator(
        train_generator, 
        steps_per_epoch=40, epochs=3,verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.01),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, epsilon=0.007)
        ],
        validation_data=val_generator, validation_steps=80, workers=4
    )
    """
    return model
#########################################################################################


from keras_squeezenet import SqueezeNet
def get_snet_layer(num_outputs=2):
    model = SqueezeNet()
    x = model.get_layer('drop9').output
    x = Convolution2D(num_outputs, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    model = Model(inputs=model.input, outputs=out)
    return model