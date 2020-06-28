# Sentiment analysis predict wrapper
import os

import numpy as np
import pandas as pd
import torch
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import tensorflow as tf

from libra.data_generation.dataset_labelmatcher import get_similar_column
from libra.data_generation.grammartree import get_value_instruction
from libra.modeling.prediction_model_creation import get_keras_text_class
from libra.plotting.generate_plots import generate_classification_plots
from libra.preprocessing.NLP_preprocessing import get_target_values, text_clean_up, lemmatize_text, encode_text
from libra.preprocessing.huggingface_model_finetune_helper import CustomDataset, train, inference
from libra.preprocessing.image_caption_helpers import load_image, map_func, CNN_Encoder, RNN_Decoder, get_path_column, \
    generate_caption_helper
from libra.query.dimensionality_red_queries import logger

def predict_text_sentiment(self, text):
    sentimentInfo = self.models.get("Text Classification LSTM")
    vocab = sentimentInfo["vocabulary"]
    # Clean up text
    text = lemmatize_text(text_clean_up([text]))
    # Encode text
    text = encode_text(vocab, text)
    text = sequence.pad_sequences(text, sentimentInfo["maxTextLength"])
    model = sentimentInfo["model"]
    prediction = tf.keras.backend.argmax(model.predict(text))
    return sentimentInfo["classes"][tf.keras.backend.get_value(prediction)[0]]


# sentiment analysis query
def text_classification_query(self, instruction,
                              preprocess=True,
                              test_size=0.2,
                              random_state=49,
                              epochs=50,
                              maxTextLength=200,
                              generate_plots=True):
    data = pd.read_csv(self.dataset)
    data.fillna(0, inplace=True)
    X, Y = get_target_values(data, instruction, "label")
    Y = np.array(Y)
    classes = np.unique(Y)

    if preprocess:
        logger("Preprocessing data...")
        X = lemmatize_text(text_clean_up(X.array))
        vocab = X
        X = encode_text(X, X)

    X = np.array(X)

    model = get_keras_text_class(maxTextLength, len(classes))

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    X_train = sequence.pad_sequences(X_train, maxlen=maxTextLength)
    X_test = sequence.pad_sequences(X_test, maxlen=maxTextLength)

    logger("Training Model...")
    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=epochs,
                        validation_split=0.1)

    logger("Testing Model...")
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=32)

    logger("->","Test accuracy:" + str(acc))

    if generate_plots:
        # generates appropriate classification plots by feeding all
        # information
        plots = generate_classification_plots(
            history, X, Y, model, X_test, y_test)

    logger("Storing information in client object...")
    # storing values the model dictionary
    self.models["Text Classification LSTM"] = {"model": model,
                                               "classes": classes,
                                               "plots": plots,
                                               "target": Y,
                                               "vocabulary": vocab,
                                               "maxTextLength": maxTextLength,
                                               'losses': {
                                                   'training_loss': history.history['loss'],
                                                   'val_loss': history.history['val_loss']},
                                               'accuracy': {
                                                   'training_accuracy': history.history['accuracy'],
                                                   'validation_accuracy': history.history['val_accuracy']}}
    return self.models["Text Classification LSTM"]


# Document summarization predict wrapper
def get_summary(self, text):
    modelInfo = self.models.get("Document Summarization")
    model = modelInfo['model']
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    MAX_LEN = 512
    SUMMARY_LEN = 150
    df = pd.DataFrame({'text': [""], 'ctext': [text]})
    set = CustomDataset(df, tokenizer, MAX_LEN, SUMMARY_LEN)
    params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0
    }
    loader = DataLoader(set, **params)
    predictions, actuals = inference(tokenizer, model, "cpu", loader)
    return predictions


# text summarization query
def summarization_query(self, instruction,
                        preprocess=True,
                        test_size=0.2,
                        random_state=49,
                        generate_plots=True):
    data = pd.read_csv(self.dataset)
    data.fillna(0, inplace=True)

    logger("Preprocessing data...")

    X, Y = get_target_values(data, instruction, "summary")
    df = pd.DataFrame({'text': Y, 'ctext': X})

    device = 'cpu'

    TRAIN_BATCH_SIZE = 64
    TRAIN_EPOCHS = 10
    LEARNING_RATE = 1e-4
    SEED = random_state
    MAX_LEN = 512
    SUMMARY_LEN = 150

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    train_size = 1 - test_size
    train_dataset = df.sample(
        frac=train_size,
        random_state=SEED).reset_index(
        drop=True)

    training_set = CustomDataset(
        train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    # used small model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=LEARNING_RATE)

    logger('Initiating Fine-Tuning for the model on your dataset...')

    for epoch in range(TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    self.models["Document Summarization"] = {
        "model": model
    }
    return self.models["Document Summarization"]


def generate_caption(self, image):
    modelInfo = self.models.get("Image Caption")
    decoder = modelInfo['decoder']
    encoder = modelInfo['encoder']
    tokenizer = modelInfo['tokenizer']
    image_features_extract_model = modelInfo['feature_extraction']
    return generate_caption_helper(image, decoder, encoder, tokenizer, image_features_extract_model)


# Image Caption Generation query
def image_caption_query(self, instruction,
                        epochs,
                        preprocess=True,
                        random_state=49,
                        generate_plots=False):
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    df = pd.read_csv(self.dataset)
    df.fillna(0, inplace=True)

    logger("Preprocessing data...")

    train_captions = []
    img_name_vector = []
    y = get_similar_column(get_value_instruction(instruction), df)
    x = get_path_column(df)

    for row in df.iterrows():
        if preprocess:
            caption = '<start> ' + row[1][y] + ' <end>'
        image_id = row[1][x]
        image_path = image_id

        img_name_vector.append(image_path)
        train_captions.append(caption)

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(sorted(set(img_name_vector)))
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    BATCH_SIZE = 1
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = top_k + 1
    num_steps = len(img_name_vector) // BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((img_name_vector, cap_vector))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    logger("Training model...")

    for epoch in range(epochs):
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss / num_steps))

    logger("Storing information in client object...")

    dir_name = os.path.dirname(img_name_vector[0])
    files = os.listdir(dir_name)

    for item in files:
        if item.endswith(".npy"):
            os.remove(os.path.join(dir_name, item))

    self.models["Image Caption"] = {
        "decoder": decoder,
        "encoder": encoder,
        "tokenizer": tokenizer,
        "feature_extraction": image_features_extract_model,
        'losses': {
            'training_loss': total_loss
        }
    }
    return self.models["Image Caption"]
