import numpy as np
import os
import cv2

from libra.preprocessing.image_preprocesser import (setwise_preprocessing,
                                                    csv_preprocessing,
                                                    classwise_preprocessing,
                                                    set_distinguisher,
                                                    already_processed,
                                                    single_class_preprocessing)
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

### train the GAN model ###
def train(combined_model, discriminator, x_train=None, epochs=10, batch_size=32):

    #Normalize input from -1 to 1
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5

    loss_discriminator_history = []
    acc_discriminator_history = []
    loss_generator_history = []

    for epoch in epochs:

        #Train generator model to generate real-looking images that the discriminator classifies as real
        noise = np.random.normal(0, 1, (batch_size, 100))
        loss_generator = combined_model.train_on_batch(noise, np.ones(batch_size))

        #First half of the batch: Generate fake images and train discriminator on them
        noise = np.random.normal(0, 1, (batch_size//2, 100))
        fake_images = combined_model.predict(noise)
        loss_discriminator_fake, acc_discriminator_fake = discriminator.train_on_batch(fake_images, np.zeros(batch_size//2))

        #Second half of the batch: Select real images from the training set uniformly at random and train discriminator on them
        idx = np.random.randint(0, len(x_train), batch_size//2)
        real_images = x_train[idx]
        loss_discriminator_real, acc_discriminator_real = discriminator.train_on_batch(real_images, np.ones(batch_size//2))

        loss_discriminator = 0.5 * np.add(loss_discriminator_fake, loss_discriminator_real)
        acc_discriminator =  0.5 * np.add(acc_discriminator_fake, acc_discriminator_real)

        loss_discriminator_history.append(loss_discriminator)
        acc_discriminator_history.append(acc_discriminator)
        loss_generator_history.append(loss_generator)

        print(f"Epoch {epoch}: [Discriminator loss: {loss_discriminator} | Discriminators Accuracy: {100 * acc_discriminator}] [Generator loss: {loss_generator}]")

    return loss_discriminator_history, acc_discriminator_history, loss_generator_history

### Use generator model to generate images (after training) ###
def generate_images(generator, num_images=10, output_path=None):
    noise = np.random.normal(0, 1, (num_images, 100))
    gen_images = generator.predict(noise)

    for i in range(num_images):
        cv2.imwrite(output_path + f"generated_image_{i}.png", gen_images[i])

def gan(instruction=None,
        num_images=None,
        preprocess=True,
        data_path=None,
        verbose=None,
        epochs=None,
        height=None,
        width=None,
        num_channels=None,
        output_path=None):

    training_path = ""

    if preprocess:
        processInfo = single_class_preprocessing(data_path=data_path)
        training_path = "/proc_training_set"

    train_images = []
    for file in os.listdir(data_path + training_path):
        if os.isfile(file):
            train_images.append(cv2.imread(file))

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

    loss_discriminator_history, acc_discriminator_history, loss_generator_history = train(model_combined, x_train=train_images, epochs=epochs, batch_size=32, verbose= verbose)
    generate_images(generator, num_images=num_images, output_path=output_path)

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
            'acc_discriminator': acc_discriminator_history
            }
        #'data_sizes': {'train_size': processInfo['train_size'], 'test_size': processInfo['test_size']}
    }
