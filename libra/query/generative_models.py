from libra.preprocessing.image_preprocesser import (setwise_preprocessing,
                                                    csv_preprocessing,
                                                    classwise_preprocessing,
                                                    set_distinguisher,
                                                    already_processed)
from libra.query.supplementaries import generate_id
from keras import Model
from keras.models import Sequential
from keras.layers import (Input, Conv2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, ZeroPadding2D, Reshape, UpSampling2D)
from keras.optimizers import Adam


### Source: https://github.com/mitchelljy/DCGAN-Keras/blob/master/DCGAN.py ###
def build_discriminator(img_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), strides=2, padding='same'))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(128, (3, 3), strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(256, (3, 3), strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

### Source: https://github.com/mitchelljy/DCGAN-Keras/blob/master/DCGAN.py ###
def build_generator(img_shape, starting_filters = 64, upsample_layers = 5, noise_shape=(100,)):
    model = Sequential()
    model.add(Dense((img_shape[0] // (2 ** upsample_layers)) *
                    (img_shape[1] // (2 ** upsample_layers)) *
                    starting_filters,
                    activation='relu',
                    input_shape=noise_shape))

    model.add(Reshape((img_shape[0] // (2 ** upsample_layers),
                       img_shape[1] // (2 ** upsample_layers),
                       starting_filters)))

    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())
    model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(3, (3, 3), padding='same', activation='tanh'))

    return model

def train(model, epochs=10, batch_size=32):


def gan(instruction=None,
        read_mode=None,
        verbose=None,
        preprocess=None,
        epochs=None,
        height=None,
        width=None,
        num_channels=None,
        output_directory):

    if preprocess:
        pass


    ### Build generator model and discriminator model
    optimizer = Adam(0.0002, 0.5)
    discriminator = build_discriminator((height, width, num_channels))
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])


    generator = build_generator()
    generator.compile(
        loss='binary_crossentropy',
        optimizer=optimizer
    )

    ### Combine the generator and discriminators into one model ###
    inp = Input(shape=100)
    img = generator(inp)

    # Freeze discriminator's weights
    discriminator.trainable = False
    valid = discriminator(img)

    model_combined = Model(inp, valid)
    train(model_combined, epochs=epochs, output_directory = output_directory)

    return {
        'id': generate_id(),
        'data_type': read_mode,
         #'data': {'train': X_train, 'test': X_test},
        'shape': (height, width, num_channels),
        "model": model_combined

        #'losses': {
        #    'training_loss': history.history['loss'],
        #    'val_loss': history.history['val_loss']},

        #'accuracy': {
        #    'training_accuracy': history.history['accuracy'],
        #    'validation_accuracy': history.history['val_accuracy']},
        #'data_sizes': {'train_size': processInfo['train_size'], 'test_size': processInfo['test_size']
        # }}
