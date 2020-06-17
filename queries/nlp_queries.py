# Sentiment analysis predict wrapper
import numpy as np
import pandas as pd
import torch
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import tensorflow as tf
from modeling.prediction_model_creation import get_keras_text_class
from plotting.generate_plots import generate_classification_plots
from preprocessing.NLP_preprocessing import get_target_values, text_clean_up, lemmatize_text, encode_text
from preprocessing.huggingface_model_finetune_helper import CustomDataset, train, inference
from queries.dimensionality_red_queries import logger


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
                              epochs=10,
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

    logger("Test accuracy:" + str(acc))

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

    logger('Initiating Fine-Tuning for the model on your dataset')

    for epoch in range(TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    self.models["Document Summarization"] = {
        "model": model
    }
    return self.models["Document Summarization"]
