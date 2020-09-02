import numpy as np
import os
import cv2
import keras.backend as K

from libra.preprocessing.image_preprocesser import (setwise_preprocessing,
                                                    csv_preprocessing,
                                                    classwise_preprocessing,
                                                    set_distinguisher,
                                                    already_processed,
                                                    single_class_preprocessing)
from libra.query.supplementaries import generate_id
from libra.query.feedforward_nn import logger, clearLog
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

    model.add(Conv2D(256, (3, 3), strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    #img = Input(shape=img_shape)
    #validity = model(img)

    #return Model(img, validity)

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

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(img_shape[2], (3, 3), padding='same', activation='tanh', name='ll'))

    #noise = Input(shape=noise_shape)
    #img = model(noise)

    #return Model(noise, img)

    return model

### train the GAN model ###
def train(combined_model, generator, discriminator, x_train=None, epochs=10, batch_size=32, verbose=1):

    #Normalize input from -1 to 1
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5

    loss_discriminator_history = []
    acc_discriminator_history = []
    loss_generator_history = []

    for epoch in range(epochs):

        #Train generator model to generate real-looking images that the discriminator classifies as real
        noise = np.random.normal(0, 1, (batch_size, 100))
        loss_generator = combined_model.train_on_batch(noise, np.ones(batch_size))

        #First half of the batch: Generate fake images and train discriminator on them
        noise = np.random.normal(0, 1, (batch_size//2, 100))
        fake_images = generator.predict(noise)
        loss_discriminator_fake, acc_discriminator_fake = discriminator.train_on_batch(fake_images, np.zeros(batch_size//2))

        #Second half of the batch: Select real images from the training set uniformly at random and train discriminator on them
        idx = np.random.randint(0, len(x_train), batch_size//2)
        real_images = x_train[idx]
        loss_discriminator_real, acc_discriminator_real = discriminator.train_on_batch(real_images, np.ones(batch_size//2))

        loss_discriminator = 0.5 * np.add(loss_discriminator_fake, loss_discriminator_real)
        acc_discriminator = 0.5 * np.add(acc_discriminator_fake, acc_discriminator_real)

        loss_discriminator_history.append(loss_discriminator)
        acc_discriminator_history.append(acc_discriminator)
        loss_generator_history.append(loss_generator)

        if verbose == 1:
            logger(f"Epoch {(epoch+1)}: [Discriminator loss: {loss_discriminator} | Discriminators Accuracy: {100 * acc_discriminator}] [Generator loss: {loss_generator}]")

    return loss_discriminator_history, acc_discriminator_history, loss_generator_history

### Use generator model to generate images (after training) ###
def generate_images(generator, num_images=3, output_path=None):
    noise = np.random.normal(0, 1, (num_images, 100))
    gen_images = generator.predict(noise)

    for i in range(num_images):
        cv2.imwrite(output_path + f"/generated_images/generated_image_{i}.jpg", gen_images[i])

### Deep Convolutional Generative Adversarial Network ###
def dcgan(instruction=None,
        num_images=None,
        preprocess=True,
        data_path=None,
        verbose=None,
        epochs=None,
        height=None,
        width=None,
        output_path=None):
    #K.clear_session()

    training_path = ""

    logger("Preprocessing images")

    num_channels = 3

    if preprocess:
        processInfo = single_class_preprocessing(data_path=data_path, height=height, width=width)
        training_path = "proc_training_set"
        num_channels = 1 if processInfo["gray_scale"] else 3

    train_images = []
    for file in os.listdir(data_path + "/" + training_path):
        abs_path = os.path.join(data_path, training_path, file)
        if os.path.isfile(abs_path):
            train_images.append(cv2.imread(abs_path))

    train_images = np.array(train_images)

    logger("Building generator model and discriminator model")

    ### Build generator model and discriminator model
    optimizer = Adam(0.0002, 0.5)

    img_shape = (processInfo["height"], processInfo["width"], num_channels)
    discriminator = build_discriminator(img_shape)
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    generator = build_generator(img_shape)
    generator.compile(
        loss='binary_crossentropy',
        optimizer=optimizer
    )

    ### Combine the generator and discriminators into one model ###

    inp = Input(shape=100)
    model_combined = Sequential()
    model_combined.add(generator)

    # Freeze discriminator's weights
    discriminator.trainable = False

    model_combined.add(discriminator)

    model_combined.compile(loss='binary_crossentropy',
                           optimizer=optimizer)

    logger("Training Generative Adversarial Network")
    loss_discriminator_history, acc_discriminator_history, loss_generator_history = train(model_combined, generator, discriminator, x_train=train_images, epochs=epochs, batch_size=32, verbose=verbose)

    logger("Generating output images")

    generate_images(generator, num_images=num_images, output_path=data_path)
    clearLog()

    K.clear_session()

    return {
        'id': generate_id(),
         'data': {'train': train_images},
        'shape': (height, width, num_channels),
        "model": model_combined,
        'losses': {
            'loss_discriminator_history': loss_discriminator_history,
            'loss_generator_history': loss_generator_history
            },

        'accuracy': {
            'acc_discriminator_history': acc_discriminator_history
            }
    }
