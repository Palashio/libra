import os
import warnings

import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from keras_preprocessing import sequence
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import TFT5ForConditionalGeneration, T5Tokenizer, \
    pipeline
import libra.plotting.nonkeras_generate_plots
from libra.data_generation.dataset_labelmatcher import get_similar_column
from libra.data_generation.grammartree import get_value_instruction
from libra.modeling.prediction_model_creation import get_keras_text_class
from libra.plotting.generate_plots import generate_classification_plots
from libra.preprocessing.NLP_preprocessing import get_target_values, text_clean_up, lemmatize_text, encode_text, \
    tokenize_for_input_ids, NoStdStreams, add_prefix
from libra.preprocessing.data_reader import DataReader
from libra.preprocessing.image_caption_helpers import load_image, map_func, CNN_Encoder, RNN_Decoder, get_path_column, \
    generate_caption_helper
from libra.query.supplementaries import save

counter = 0

currLog = 0

warnings.filterwarnings("ignore")


def clearLog():
    global currLog
    global counter

    currLog = ""
    counter = 0


def logger(instruction, found=""):
    '''
    logging function that creates hierarchial display of the processes of
    different functions. Copied into different python files to maintain
    global variables.

    :param instruction: what you want to be displayed
    :param found: if you want to display something found like target column
    '''

    global counter
    if counter == 0:
        print((" " * 2 * counter) + str(instruction) + str(found))
    elif instruction == "->":
        counter = counter - 1
        print(Fore.BLUE + (" " * 2 * counter) +
              str(instruction) + str(found) + (Style.RESET_ALL))
    else:
        print((" " * 2 * counter) + "|- " + str(instruction) + str(found))
        if instruction == "done...":
            print("\n" + "\n")

    counter += 1


# Sentiment analysis predict wrapper
def classify_text(self, text):
    """
    function to perform sentiment analysis text_classification

    :param text: text sent in/from written query to be analyzed
    """

    sentimentInfo = self.models.get("text_classification")
    vocab = sentimentInfo["vocabulary"]
    # Clean up text
    text = lemmatize_text(text_clean_up([text]))
    # Encode text
    text = encode_text(vocab, text)
    text = sequence.pad_sequences(text, sentimentInfo["max_text_length"])
    model = sentimentInfo["model"]
    prediction = tf.keras.backend.argmax(model.predict(text))
    return sentimentInfo["classes"][tf.keras.backend.get_value(prediction)[0]]


# Sentiment analysis query
def text_classification_query(self, instruction, drop=None,
                              preprocess=True,
                              label_column=None,
                              test_size=0.2,
                              random_state=49,
                              learning_rate=1e-2,
                              epochs=5,
                              monitor="val_loss",
                              batch_size=32,
                              max_text_length=20,
                              generate_plots=True,
                              save_model=False,
                              save_path=os.getcwd()):
    """
    function to apply text_classification algorithm for sentiment analysis
    :param many params: used to hyperparametrize the function.
    :return a dictionary object with all of the information for the algorithm.
    """

    if test_size < 0:
        raise Exception("Test size must be a float between 0 and 1")

    if test_size >= 1:
        raise Exception(
            "Test size must be a float between 0 and 1 (a test size greater than or equal to 1 results in no training "
            "data)")

    if epochs < 1:
        raise Exception("Epoch number is less than 1 (model will not be trained)")

    if max_text_length <= 1:
        raise Exception("Max text length should be larger than 1")

    if batch_size < 1:
        raise Exception("Batch size must be equal to or greater than 1")

    if save_model:
        if not os.path.exists(save_path):
            raise Exception("Save path does not exists")

    if test_size == 0:
        testing = False
    else:
        testing = True

    data = DataReader(self.dataset)
    data = data.data_generator()

    if preprocess:
        data.fillna(0, inplace=True)

    if drop is not None:
        data.drop(drop, axis=1, inplace=True)

    if label_column is None:
        label = "label"
    else:
        label = label_column

    X, Y, target = get_target_values(data, instruction, label)
    Y = np.array(Y)
    classes = np.unique(Y)

    logger("->", "Target Column Found: {}".format(target))

    vocab = {}
    if preprocess:
        logger("Preprocessing data")
        X = lemmatize_text(text_clean_up(X.array))
        vocab = X
        X = encode_text(X, X)

    X = np.array(X)
    model = get_keras_text_class(len(vocab), len(classes), learning_rate)
    logger("Building Keras LSTM model dynamically")

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)

    X_train = sequence.pad_sequences(X_train, maxlen=max_text_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_text_length)

    y_vals = np.unique(np.append(y_train, y_test))
    label_mappings = {}
    for i in range(len(y_vals)):
        label_mappings[y_vals[i]] = i
    map_func = np.vectorize(lambda x: label_mappings[x])
    y_train = map_func(y_train)
    y_test = map_func(y_test)

    logger("Training initial model")

    # early stopping callback
    es = EarlyStopping(
        monitor=monitor,
        mode='auto',
        verbose=0,
        patience=5)


    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            batch_size=batch_size,
                            epochs=epochs, callbacks=[es], verbose=0)

    logger("->", "Final training loss: {}".format(history.history["loss"][len(history.history["loss"]) - 1]))
    if testing:
        logger("->",
               "Final validation loss: {}".format(history.history["val_loss"][len(history.history["val_loss"]) - 1]))
        logger("->", "Final validation accuracy: {}".format(
            history.history["val_accuracy"][len(history.history["val_accuracy"]) - 1]))
        losses = {'training_loss': history.history['loss'], 'val_loss': history.history['val_loss']}
        accuracy = {'training_accuracy': history.history['accuracy'],
                    'validation_accuracy': history.history['val_accuracy']}
    else:
        logger("->", "Final validation loss: {}".format("0, No validation done"))
        losses = {'training_loss': history.history['loss']}
        accuracy = {'training_accuracy': history.history['accuracy']}

    plots = {}
    if generate_plots:
        # generates appropriate classification plots by feeding all
        # information
        logger("Generating plots")
        plots = generate_classification_plots(history)

    if save_model:
        save(model, save_model, save_path=save_path)

    logger("Storing information in client object under key 'text_classification'")
    # storing values the model dictionary

    self.models["text_classification"] = {"model": model,
                                          "classes": classes,
                                          "plots": plots,
                                          "target": Y,
                                          "vocabulary": vocab,
                                          "interpreter": label_mappings,
                                          # "max_text_length": max_text_length,
                                          'test_data': {'X': X_test, 'y': y_test},
                                          'losses': losses,
                                          'accuracy': accuracy}
    clearLog()
    return self.models["text_classification"]


# Summarization predict wrapper
def get_summary(self, text, max_summary_length=50, num_beams=4, no_repeat_ngram_size=2, num_return_sequences=1,
                early_stopping=True):
    modelInfo = self.models.get("summarization")
    model = modelInfo['model']
    tokenizer = modelInfo['tokenizer']
    text = [text]
    text = add_prefix(text, "summarize: ")
    result = model.generate(
        tf.convert_to_tensor(tokenize_for_input_ids(text, tokenizer, max_length=modelInfo['max_text_length'])),
        max_length=max_summary_length, num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size, num_return_sequences=num_return_sequences,
        early_stopping=early_stopping)
    return [tokenizer.decode(summary) for summary in result]


# Text summarization query
def summarization_query(self, instruction, preprocess=True, label_column=None,
                        drop=None,
                        epochs=5,
                        batch_size=32,
                        learning_rate=3e-5,
                        max_text_length=512,
                        gpu=False,
                        test_size=0.2,
                        random_state=49,
                        generate_plots=True,
                        save_model=False,
                        save_path=os.getcwd()):
    '''
    function to apply algorithm for text summarization
    :param many params: used to hyperparametrize the function.
    :return a dictionary object with all of the information for the algorithm.
    '''

    if test_size < 0:
        raise Exception("Test size must be a float between 0 and 1")

    if test_size >= 1:
        raise Exception(
            "Test size must be a float between 0 and 1 (a test size greater than or equal to 1 results in no training "
            "data)")

    if max_text_length < 2:
        raise Exception("Text and summary must be at least of length 2")

    if epochs < 1:
        raise Exception("Epoch number is less than 1 (model will not be trained)")

    if batch_size < 1:
        raise Exception("Batch size must be equal to or greater than 1")

    if max_text_length < 1:
        raise Exception("Max text length must be equal to or greater than 1")

    if save_model:
        if not os.path.exists(save_path):
            raise Exception("Save path does not exist")

    if test_size == 0:
        testing = False
    else:
        testing = True

    if gpu:
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            raise Exception("Please install GPU version of Tensorflow")

        device = '/device:GPU:0'
    else:
        device = '/device:CPU:0'

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    data = DataReader(self.dataset)
    data = data.data_generator()

    if drop is not None:
        data.drop(drop, axis=1, inplace=True)

    if preprocess:
        data.fillna(0, inplace=True)

    logger("Preprocessing data...")

    if label_column is None:
        label = "summary"
    else:
        label = label_column
    with NoStdStreams():
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # Find target columns
    X, Y, target = get_target_values(data, instruction, label)
    logger("->", "Target Column Found: {}".format(target))
    logger("Establishing dataset walkers")

    # Clean up text
    if preprocess:
        logger("Preprocessing data")
        X = add_prefix(lemmatize_text(text_clean_up(X.array)), "summarize: ")
        Y = add_prefix(lemmatize_text(text_clean_up(Y.array)), "summarize: ")

    # tokenize text/summaries
    X = tokenize_for_input_ids(X, tokenizer, max_text_length)
    Y = tokenize_for_input_ids(Y, tokenizer, max_text_length)

    logger('Fine-Tuning the model on your dataset...')

    # Suppress unnecessary output
    with NoStdStreams():
        model = TFT5ForConditionalGeneration.from_pretrained("t5-small", output_loading_info=False)

    if testing:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(10000).batch(batch_size)
    else:
        X_train = X
        y_train = Y
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    total_training_loss = []
    total_validation_loss = []

    # Training Loop
    with tf.device(device):
        for epoch in range(epochs):
            total_loss = 0
            total_loss_val = 0
            for data, truth in train_dataset:
                with tf.GradientTape() as tape:
                    out = model(inputs=data, decoder_input_ids=data)
                    loss_value = loss(truth, out[0])
                    total_loss += loss_value
                    grads = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

            total_training_loss.append(total_loss)

            # Validation Loop
            if testing:
                for data, truth in test_dataset:
                    logits = model(inputs=data, decoder_input_ids=data, training=False)
                    val_loss = loss(truth, logits[0])
                    total_loss_val += val_loss

                total_validation_loss.append(total_loss_val)

    logger("->", "Final training loss: {}".format(str(total_training_loss[len(total_training_loss) - 1].numpy())))

    if testing:
        total_loss_val_str = str(total_validation_loss[len(total_validation_loss) - 1].numpy())
    else:
        total_loss_val = [0]
        total_loss_val_str = str("0, No validation done")

    logger("->", "Final validation loss: {}".format(total_loss_val_str))

    if testing:
        losses = {"Training loss": total_training_loss[len(total_training_loss) - 1].numpy(),
                  "Validation loss": total_validation_loss[len(total_validation_loss) - 1].numpy()}
    else:
        losses = {"Training loss": total_training_loss[len(total_training_loss) - 1].numpy()}

    plots = None
    if generate_plots:
        logger("Generating plots")
        plots = {"loss": libra.plotting.nonkeras_generate_plots.plot_loss(total_training_loss, total_validation_loss)}

    if save_model:
        logger("Saving model")
        model.save_weights(save_path + "summarization_checkpoint.ckpt")

    logger("Storing information in client object under key 'summarization'")

    self.models["summarization"] = {
        "model": model,
        "max_text_length": max_text_length,
        "plots": plots,
        "tokenizer": tokenizer,
        'losses': losses}

    clearLog()
    return self.models["summarization"]


# image_caption Generation Prediction
def generate_caption(self, image):
    '''
    wrapper function of image_caption_query to predict caption for image
    :param image: image to be analyzed
    '''
    modelInfo = self.models.get("image_caption")
    decoder = modelInfo['decoder']
    encoder = modelInfo['encoder']
    tokenizer = modelInfo['tokenizer']
    image_features_extract_model = modelInfo['feature_extraction']
    return generate_caption_helper(
        image,
        decoder,
        encoder,
        tokenizer,
        image_features_extract_model)


# image_caption Generation query
def image_caption_query(self, instruction, label_column=None,
                        drop=None,
                        epochs=10,
                        preprocess=True,
                        random_state=49,
                        test_size=0.2,
                        top_k=5000,
                        batch_size=32,
                        buffer_size=1000,
                        embedding_dim=256,
                        units=512,
                        gpu=False,
                        generate_plots=True,
                        save_model_decoder=False,
                        save_path_decoder=os.getcwd(),
                        save_model_encoder=False,
                        save_path_encoder=os.getcwd()):
    '''
    function to apply predictive algorithm for image_caption generation
    :param many params: used to hyperparametrize the function.
    :return a dictionary object with all of the information for the algorithm.
    '''

    if test_size < 0:
        raise Exception("Test size must be a float between 0 and 1")

    if test_size >= 1:
        raise Exception(
            "Test size must be a float between 0 and 1 (a test size greater than or equal to 1 results in no training "
            "data)")

    if top_k < 1:
        raise Exception("Top_k value must be equal to or greater than 1")

    if batch_size < 1:
        raise Exception("Batch size must be equal to or greater than 1")

    if buffer_size < 1:
        raise Exception("Buffer size must be equal to or greater than 1")

    if embedding_dim < 1:
        raise Exception("Embedding dimension must be equal to or greater than 1")

    if units < 1:
        raise Exception("Units must be equal to or greater than 1")

    if epochs < 1:
        raise Exception("Epoch number is less than 1 (model will not be trained)")

    if save_model_decoder:
        if not os.path.exists(save_path_decoder):
            raise Exception("Decoder save path does not exists")

    if save_model_encoder:
        if not os.path.exists(save_path_encoder):
            raise Exception("Encoder save path does not exists")

    if test_size == 0:
        testing = False
    else:
        testing = True

    if gpu:
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            raise Exception("Please install GPU version of Tensorflow")

        device = '/device:GPU:0'
    else:
        device = '/device:CPU:0'

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    data = DataReader(self.dataset)
    df = data.data_generator()

    if preprocess:
        df.fillna(0, inplace=True)
    if drop is not None:
        df.drop(drop, axis=1, inplace=True)

    logger("Preprocessing data")

    train_captions = []
    img_name_vector = []

    if label_column is None:
        label = instruction
    else:
        label = label_column

    x = get_path_column(df)
    y = get_similar_column(get_value_instruction(label), df)
    logger("->", "Target Column Found: {}".format(y))

    for row in df.iterrows():
        if preprocess:
            caption = '<start> ' + row[1][y] + ' <end>'
        image_id = row[1][x]
        image_path = image_id

        img_name_vector.append(image_path)
        train_captions.append(caption)
    with NoStdStreams():
        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    logger("Extracting features from model")
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    image_dataset = tf.data.Dataset.from_tensor_slices(
        sorted(set(img_name_vector)))
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(
            batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())
    logger("->", "Tokenizing top {} words".format(top_k))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding='post')

    vocab_size = top_k + 1
    # num_steps = len(img_name_vector) // batch_size

    if testing:
        img_name_train, img_name_val, cap_train, cap_val = train_test_split(
            img_name_vector, cap_vector, test_size=test_size, random_state=0)
    else:
        img_name_train = img_name_vector
        cap_train = cap_vector

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    logger("Shuffling dataset")
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if testing:
        dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))

        dataset_val = dataset_val.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32]),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Shuffle and batch
        dataset_val = dataset_val.shuffle(buffer_size).batch(batch_size)
        dataset_val = dataset_val.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

    logger("Establishing encoder decoder framework")
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
        with tf.device(device):
            loss = 0

            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden = decoder.reset_state(batch_size=target.shape[0])

            dec_input = tf.expand_dims(
                [tokenizer.word_index['<start>']] * target.shape[0], 1)

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

    @tf.function
    def val_step(img_tensor, target):
        with tf.device(device):
            loss = 0

            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden = decoder.reset_state(batch_size=target.shape[0])

            dec_input = tf.expand_dims(
                [tokenizer.word_index['<start>']] * target.shape[0], 1)

            with tf.GradientTape() as tape:
                features = encoder(img_tensor)

                for i in range(1, target.shape[1]):
                    # passing the features through the decoder
                    predictions, hidden, _ = decoder(dec_input, features, hidden)

                    loss += loss_function(target[:, i], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(target[:, i], 1)

            total_loss = (loss / int(target.shape[1]))
            return total_loss

    logger("Training model...")
    with tf.device(device):
        loss_plot_train = []
        loss_plot_val = []
        for epoch in range(epochs):
            total_loss = 0
            total_loss_val = 0

            for (batch, (img_tensor, target)) in enumerate(dataset):
                batch_loss, t_loss = train_step(img_tensor, target)
                total_loss += t_loss

            loss_plot_train.append(total_loss.numpy())

            if testing:
                for (batch, (img_tensor, target)) in enumerate(dataset_val):
                    batch_loss, t_loss = train_step(img_tensor, target)
                    total_loss_val += t_loss

                loss_plot_val.append(total_loss_val.numpy())

    dir_name = os.path.dirname(img_name_vector[0])
    files = os.listdir(dir_name)

    for item in files:
        if item.endswith(".npy"):
            os.remove(os.path.join(dir_name, item))

    plots = {}
    if generate_plots:
        logger("Generating plots")
        plots.update({"loss": libra.plotting.nonkeras_generate_plots.plot_loss(loss_plot_train, loss_plot_val)})

    logger("->", "Final training loss: {}".format(str(total_loss.numpy())))
    total_loss = total_loss.numpy()
    if testing:
        total_loss_val = total_loss_val.numpy()
        total_loss_val_str = str(total_loss_val)
    else:
        total_loss_val = 0
        total_loss_val_str = str("0, No validation done")

    logger("->", "Final validation loss: {}".format(total_loss_val_str))

    if save_model_decoder:
        logger("Saving decoder checkpoint...")
        encoder.save_weights(save_path_decoder + "decoderImgCap.ckpt")

    if save_model_encoder:
        logger("Saving encoder checkpoint...")
        encoder.save_weights(save_path_encoder + "encoderImgCap.ckpt")

    logger("Storing information in client object under key 'image_caption'")

    self.models["image_caption"] = {
        "decoder": decoder,
        "encoder": encoder,
        "tokenizer": tokenizer,
        "feature_extraction": image_features_extract_model,
        "plots": plots,
        'losses': {
            'Training loss': total_loss,
            'Validation loss': total_loss_val
        }
    }
    clearLog()
    return self.models["image_caption"]


def generate_text(self, prefix=None,
                  file_data=True,
                  max_length=512,
                  do_sample=True,
                  top_k=50,
                  top_p=0.9,
                  temperature=0.3,
                  return_sequences=2):
    '''
    Takes in initial text and generates text with specified number of characters more using Top P sampling
    :param prefix: initial text to start with
    :param several parameters to hyperparemeterize with given defaults
    :return: complete generated text
    '''

    if return_sequences < 1:
        raise Exception("return sequences number is less than 1 (need an integer of atleast 1)")

    if max_length < 1:
        raise Exception("Max text length must be equal to or greater than 1")

    with NoStdStreams():
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    if file_data:
        f = open(self.dataset, "r")
        input_ids = tokenizer.encode(f.read(), return_tensors='tf', max_length=max_length - 1, truncation=True)
        f.close()
    else:
        input_ids = tokenizer.encode(prefix, return_tensors='tf', max_length=max_length - 1, truncation=True)

    logger("Generating text now...")
    tf.random.set_seed(0)
    output = model.generate(input_ids,
                            do_sample=do_sample,
                            max_length=max_length,
                            top_k=top_k, top_p=top_p,
                            temperature=temperature,
                            num_return_sequences=return_sequences)
    total_text = ""
    for i, sample_output in enumerate(output):
        value = "{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True))
        total_text += value

    self.models['text_generation'] = {"generated_text": total_text}
    return self.models['text_generation']


# name entity recognition query
def get_ner(self, instruction):
    """
    function to identify name entities
    :param instruction: Used to get target column
    :return: dictionary object with detected name-entities
    """
    data = DataReader(self.dataset)
    data = data.data_generator()

    target = get_similar_column(get_value_instruction(instruction), data)
    logger("->", "Target Column Found: {}".format(target))

    # Remove stopwords if any from the detection column
    data['combined_text_for_ner'] = data[target].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stopwords.words()]))

    logger("Detecting Name Entities from : {} data files".format(data.shape[0] + 1))

    # Named entity recognition pipeline, default model selection
    with NoStdStreams():
        hugging_face_ner_detector = pipeline('ner', grouped_entities=True, framework='tf')
        data['ner'] = data['combined_text_for_ner'].apply(lambda x: hugging_face_ner_detector(x))
    logger("NER detection status complete")
    logger("Storing information in client object under key 'named_entity_recognition'")

    self.models["named_entity_recognition"] = {
        "model": hugging_face_ner_detector.model,
        "tokenizer": hugging_face_ner_detector.tokenizer,
        'name_entities': data['ner'].to_dict()}

    logger("Output: ", data['ner'].to_dict())
    clearLog()
    return self.models["named_entity_recognition"]
