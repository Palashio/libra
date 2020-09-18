# def stats(
#     dataset=None,
#     drop=None,
#     column_name=None,
# ):
#     logger("Reading in dataset....")
#     # Reading in dataset and creating pdtabulate variable to format outputs
#     dataReader = DataReader(dataset)
#     data = dataReader.data_generator()

#     if drop is not None:
#         data.drop(drop, axis=1, inplace=True)

#     data.fillna(0, inplace=True)
#     logger("Creating lambda object to format...")

#     def pdtabulate(df):
#         return tabulate(
#             df, headers='keys', tablefmt='psql')

#     logger("Identifying columns to transform....")

#     # identifying categorical and numerical columns, and encoding
#     # appropriately
#     categor = data.select_dtypes(exclude=['int', 'float'])
#     categor = categor.apply(LabelEncoder().fit_transform)
#     for value in categor.columns:
#         data[str(value)] = categor[str(value)]

#     # if user doesn't specify column analysis on performed on the whole
#     # dataset
#     if column_name == "none":
#         columns = []
#         sim = []
#         for first_val in data.columns:
#             for sec_val in data.columns:
#                 if first_val == sec_val:
#                     continue
#                 columns.append(str(first_val) + "_" + str(sec_val))
#                 sim.append(1 - cosine(data[first_val], data[sec_val]))
#             df = pd.DataFrame(columns=columns)
#             df.loc[len(df)] = sim

#             cols = []
#             vals = []
#             logger("Restructuring dataset for similarity...")
#             # identifying top 5 feature importances and appending them to an
#             # array for display
#             for val in np.argpartition(np.asarray(df.iloc[0]), -5)[-5:]:
#                 cols.append(df.columns[val])
#                 vals.append(df[df.columns[val]].iloc[0])
#                 frame = pd.DataFrame(columns=cols)
#                 frame.loc[len(df)] = vals
#             print("Similarity Spectrum")
#             print(pdtabulate(frame))
#             print()
#             print("Dataset Description")
#             print(pdtabulate(data.describe()))

#         else:
#             logger("Performing similarity calculations....")
#             columns = []
#             sim = []
#             # identifying columns to be compared
#             for val in data.columns:
#                 if val == column_name:
#                     continue
#                 columns.append(str(column_name) + "_" + str(val))
#                 sim.append(1 - cosine(data[column_name], data[val]))
#             df = pd.DataFrame(columns=columns)
#             df.loc[len(df)] = sim

#         cols = []
#         vals = []
#         # identifying top 5 feature importances and appending them to a
#         # dataset
#         for val in np.argpartition(np.asarray(df.iloc[0]), -5)[-5:]:
#             cols.append(df.columns[val])
#             vals.append(df[df.columns[val]].iloc[0])
#             frame = pd.DataFrame(columns=cols)
#             frame.loc[len(df)] = vals

#         # displaying the similarity spectrum and the formatted
#         # data.describe()
#     print("Similarity Spectrum")
#     print("-------------------------")
#     print(pdtabulate(frame))
#     print()
#     print("Dataset Description")
#     print("-------------------------")
#     print(pdtabulate(data[column_name]).describe())


#def booster(instruction,dataset,y,target="",obj):
#    #obj=["reg:linear","multi:softmax "]
#    if target == "":
#        y=data_y(instruction)
#
#    X_train, X_test, y_train, y_test = train_test_split(
#    dataset, y, test_size=0.2, random_state=49)
#    clf = XGBClassifier(objective=obj,learning_rate =0.1,silent=1,alpha = 10)
#    clf.fit(X_train, y_train)
#    y_pred=clf.predict(X_test)
#    return accuracy_score(y_pred, y_test), y_pred 
#    #importance graph
#   feature_important = model.get_booster().get_score(importance_type='weight')
#   keys = list(feature_important.keys())
#   values = list(feature_important.values())
#
#   data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
#   data.plot(kind='barh')


#Squeezenet
"""
from keras_squeezenet import SqueezeNet
from keras.models import Model
from keras.layers import  Convolution2D, MaxPooling2D, Lambda
from keras.layers import Activation, GlobalAveragePooling2D, concatenate
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras import backend
from keras import optimizers

temperature, lambda_const = 5.0, 0.2
num_classes=2
def knowledge_distillation_loss(y_true, y_pred, lambda_const,num_classes):
    y_true, logits = y_true[:, :num_classes], y_true[:, num_classes:]
    y_soft = backend.softmax(logits/temperature)
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

def get_snet_layer(num_out=2):
    global num_outputs
    num_outputs=num_out
    model = SqueezeNet()

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
    model.compile(
        optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True), 
        loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const), 
        metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss]
    )
    return model

model= get_snet_layer(num_classes)
history = model.fit_generator(
        X_train, 
        steps_per_epoch=X_train.n //
        X_train.batch_size, epochs=25,verbose=0,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.01),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, epsilon=0.007)
        ],
        validation_data=X_test, validation_steps=X_test.n //
        X_test.batch_size, workers=4
    )
"""









# # Checks columns for text
# # Converts text to numbers using TF-IDF
# def process_text(data):
#
#     combined = pd.concat([data['train'], data['test']], axis=0)
#
#     combined.dropna(inplace=True)
#
#     possible_text_cols = combined.select_dtypes(
#         exclude=["number"]).columns
#     text_cols = list()
#
#     # Scans categorical columns for text
#     for col in possible_text_cols:
#         print(len(pd.unique(combined[col])))
#         print(len(combined))
#         print("\n\n")
#         if len(np.unique(combined[col])) == len(combined):
#             text_cols.append(col)
#
#     print(text_cols)
#     #text_preprocessing()


# Returns True if text has more than 3 spaces
# Otherwise false
# def find_spaces(text):
#     spaces = 0
#     for i in text:
#         if spaces > 3:
#             return True
#         if i.isspace():
#             spaces += 1
#     return False



        # trainingImages = []
        # train_labels = []
        # validationImages = []
        # test_labels = []
        #
        # for path in imgPaths:
        # classLabel = path.split(os.path.sep)[-2]
        # classes.add(classLabel)
        # img = img_to_array(load_img(path, target_size=(64, 64)))
        #
        # if path.split(os.path.sep)[-3] == 'training_set':
        #     trainingImages.append(img)
        #     train_labels.append(classLabel)
        # else:
        #     validationImages.append(img)
        #     test_labels.append(classLabel)
        #
        # trainingImages = np.array(trainingImages)
        # train_labels = to_categorical(np.array(train_labels))
        # validationImages = np.array(validationImages)
        # test_labels = to_categorical(np.array(test_labels))
        # model.compile(loss=’categorical_crossentropy’,
        #           optimizer=’sgd’,
        #           metrics=[‘accuracy’])
        # history=model.fit(train_images, train_labels,
        #           batch_size=100,
        #           epochs=5,
        #           verbose=1)



# Seperates the color channels and then reshapes each of the channels to
# (224, 224)
# def processColorChanel(img):
#     b, g, r = cv2.split(img)
#     # seperating each value into a color channel and resizing to a standard
#     # size of 224, 224, 3 <- because of RGB color channels. If it's not 3
#     # color channels it'll pad with zeroes
#     b = cv2.resize(b, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
#     g = cv2.resize(g, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
#     r = cv2.resize(r, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
#     img = cv2.merge((b, g, r))
#     return img

# # returns metrics about your dataset including similarity information
# def stat_analysis(self, column_name="none", drop=None):
#     stats(
#         dataset=self.dataset,
#         drop=drop,
#         column_name=column_name
#     )
#
#     return



### TUNER UPDATE
# def tuneReg(
#         data,
#         target,
#         max_layers=10,
#         min_layers=2,
#         min_dense=32,
#         max_dense=512,
#         executions_per_trial=3,
#         max_trials=3,
#         epochs=10,
#         activation='relu',
#         step=32,
#         verbose=0,
#         test_size=0.2
# ):
#
#     # function build model using hyperparameter
#     def build_model(hp):
#         model = mlrose.NeuralNetwork( hidden_nodes = [ hp.Int('num_layers', min_layers,
#                                                        max_layers)],
#                                       activation = activation,
#                                       algorithm = 'random_hill_climb',
#                                       max_iters = 1000,
#                                       bias = True, learning_rate = 0.0001,
#                                       early_stopping = True, clip_max = 5,
#                                       max_attempts = 100,
#                                       random_state = 3)
#         model.compile(
#             optimizer=keras.optimizers.Adam(
#                                        hp.Float('learning_rate',
#                                                 min_value=1e-5,
#                                                 max_value=1e-2,
#                                                 sampling='LOG',
#                                                 default=1e-3)),
#             loss='mean_squared_error',
#             metrics=[metrics])
#         return model
#
#     # Create regularization hyperparameter distribution
#     create_regularizer = uniform(loc=0, scale=4)
#     # Create hyperparameter options
#     hyperparameters = dict(C=create_regularizer, penalty=['l1', 'l2'])
#     # random search for the model
#     tuner = RandomizedSearchCV( build_model,
#                                 hyperparameters,
#                                 random_state=1,
#                                 n_iter=100,
#                                 cv=5, verbose=0,
#                                 n_jobs=-1)
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         data, target, test_size=0.2, random_state=49)
#     history = tuner.fit( X_train,
#                          y_train,
#                          epochs=epochs,
#                          validation_data=(
#                                  X_test,
#                                  y_test),
#                          callbacks= [tf.keras.callbacks.TensorBoard('my_dir')],
#                          verbose=0)
#     """
#     #Return:
#     #    models[0] : best model obtained after tuning
#     #    best_hps : best Hyperprameters obtained after tuning, stored as map
#     #    history : history of the data executed from the given model
#     """
#     return tuner,best_model.bestestimator_.get_params()['C'], history
#

# def tuneClass(
# #         X,
# #         y,
# #         num_classes,
# #         max_layers=10,
# #         min_layers=2,
# #         min_dense=32,
# #         max_dense=512,
# #         executions_per_trial=3,
# #         max_trials=3,
# #         activation='relu',
# #         loss='categorical_crossentropy',
# #         metrics='accuracy',
# #         epochs=10,
# #         step=32,
# #         verbose=0,
# #         test_size=0.2):
# #     # function build model using hyperparameter
# #     le = preprocessing.LabelEncoder()
# #     y = tf.keras.utils.to_categorical(
# #         le.fit_transform(y), num_classes=num_classes)
# #
# #     def build_model(hp):
# #         model = mlrose.NeuralNetwork( hidden_nodes = [hp.Int('num_layers', min_layers, max_layers)],
# #                                           activation = activation,
# #                                           algorithm = 'random_hill_climb',
# #                                           max_iters = 1000,
# #                                           bias = True, is_classifier = True,
# #                                           learning_rate = 0.0001,
# #                                           early_stopping = True, clip_max = 5,
# #                                           max_attempts = 100,
# # 				                          random_state = 3)
# #         model.compile(
# #             optimizer=keras.optimizers.Adam(
# #                                        hp.Float('learning_rate',
# #                                                 min_value=1e-5,
# #                                                 max_value=1e-2,
# #                                                 sampling='LOG',
# #                                                 default=1e-3)),
# #             loss=loss,
# #             metrics=[metrics])
# #         return model
# #
# #     # Create regularization hyperparameter distribution
# #     create_regularizer = uniform(loc=0, scale=4)
# #     # Create hyperparameter options
# #     hyperparameters = dict(C=create_regularizer, penalty=['l1', 'l2'])
# #     # random search for the model
# #     tuner = RandomizedSearchCV( build_model,
# #                                 hyperparameters,
# #                                 random_state=1,
# #                                 n_iter=100,
# #                                 cv=5, verbose=0,
# #                                 n_jobs=-1)
# #     # tuners, establish the object to look through the tuner search space
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.2, random_state=49)
# #     history = tuner.fit( X_train,
# #                          y_train,
# #                          epochs=epochs,
# #                          validation_data=(
# #                                  X_test,
# #                                  y_test),
# #                          callbacks= [tf.keras.callbacks.TensorBoard('my_dir')],
# #                          verbose=0)
# #     """
# #     #Return:
# #     #    models[0] : best model obtained after tuning
# #     #    best_hps : best Hyperprameters obtained after tuning, stored as array
# #     #    history : history of the data executed from the given model
# #     """
# #     return tuner,best_model.bestestimator_.get_params()['C'], history
# #     #return models[0], hyp, history
# #
# # """"""
    
#     
# def get_standard_training_output_keras(epochs, history):
#     '''
#     helper output for logger
#     :param epochs: is the number of epochs model was running for
#     :param history: the keras history object
#     '''
#     global counter
#     col_name = [["Epochs", "| Training Loss ", "| Validation Loss "]]
#     col_width = max(len(word) for row in col_name for word in row) + 2
#     for row in col_name:
#         print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
#                                                     for word in row)) + " |")
# 
#     for i, j, k in zip(
#             range(epochs), history.history["loss"], history.history["val_loss"]):
#         values = []
#         values.append(str(i))
#         values.append("| " + str(j))
#         values.append("| " + str(k))
#         datax = []
#         datax.append(values)
#         for row in datax:
#             print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
#                                                         for word in row)) + " |")
# 
# 
# def get_standard_training_output_generic(epochs, loss, val_loss):
#     '''
#     helper output for logger
#     :param epochs: is the number of epochs model was running for
#     :param loss: is the amount of loss in the training instance
#     :param val_loss: just validation loss
#     '''
#     global counter
#     col_name = [["Epochs ", "| Training Loss ", "| Validation Loss "]]
#     col_width = max(len(word) for row in col_name for word in row) + 2
#     for row in col_name:
#         print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
#                                                     for word in row)) + " |")
# 
#     for i, j, k in zip(range(epochs), loss, val_loss):
#         values = []
#         values.append(str(i))
#         values.append("| " + str(j))
#         values.append("| " + str(k))
#         datax = []
#         datax.append(values)
#         for row in datax:
#             print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
#                                                         for word in row)) + " |") 
# def booster(dataset, obj):
#     # obj=["reg:linear","multi:softmax "]
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         dataset, y, test_size=0.2, random_state=49)
#     clf = XGBClassifier(
#         objective=obj,
#         learning_rate=0.1,
#         silent=1,
#         alpha=10)
#     clf.fit(X_train, y_train)
#     return accuracy_score(clf.predict(X_test_mod), y_test_mod)
#     # importance graph
#     # plt.rcParams['figure.figsize'] = [5, 5]
#     # plt.show()}


# def dimensionality_reduc(
#         instruction,
#         dataset,
#         arr=[
#             "RF",
#             "PCA",
#             "KPCA",
#             "ICA"],
#         inplace=False):
#     global counter
#
#     dataReader = DataReader(dataset)
#
#     logger("loading dataset...")
#     data = dataReader.data_generator()
#     data.fillna(0, inplace=True)
#
#     logger("getting most similar column from instruction...")
#     target = get_similar_column(get_value_instruction(instruction), data)
#
#     y = data[target]
#     del data[target]
#     le = preprocessing.LabelEncoder()
#     y = le.fit_transform(y)
#
#     data = structured_preprocesser(data)
#
#     perms = []
#     overall_storage = []
#     finals = []
#
#     logger("generating dimensionality permutations...")
#     for i in range(1, len(arr) + 1):
#         for elem in list(permutations(arr, i)):
#             perms.append(elem)
#
#     logger("running each possible permutation...")
#     logger("realigning tensors...")
#     for path in perms:
#         currSet = data
#         for element in path:
#             if element == "RF":
#                 data_mod, beg_acc, final_acc, col_removed = dimensionality_RF(
#                     instruction, currSet, target, y)
#             elif element == "PCA":
#                 data_mod, beg_acc, final_acc, col_removed = dimensionality_PCA(
#                     instruction, currSet, target, y)
#             elif element == "KPCA":
#                 data_mod, beg_acc, final_acc, col_removed = dimensionality_KPCA(
#                     instruction, currSet, target, y)
#             elif element == "ICA":
#                 data_mod, beg_acc, final_acc, col_removed = dimensionality_ICA(
#                     instruction, currSet, target, y)
#             overall_storage.append(
#                 list([data_mod, beg_acc, final_acc, col_removed]))
#             currSet = data_mod
#         finals.append(overall_storage[len(overall_storage) - 1])
#
#     logger("Fetching Best Accuracies...")
#     accs = []
#     logger("->", "Baseline Accuracy: " + str(finals[0][1]))
#     # print("----------------------------")
#     col_name = [["Permutation ", "| Final Accuracy "]]
#     printtable(col_name, max(len(word)
#                              for row in col_name for word in row) + 5)
#     for i, element in product(range(len(finals)), finals):
#         values = []
#         values.append(str(perms[i]))
#         values.append("| " + str(element[2]))
#         datax = []
#         datax.append(values)
#         printtable(datax, max(len(word)
#                               for row in col_name for word in row) + 5)
#         del values, datax
#         if finals[0][1] < element[2]:
#             accs.append(list([str(perms[i]),
#                               "| " + str(element[2])]))
#     print("")
#     logger("->", " Best Accuracies")
#     # print("----------------------------")
#     col_name = [["Permutation ", "| Final Accuracy "]]
#     printtable(col_name, max(len(word)
#                              for row in col_name for word in row) + 5)
#     printtable(accs, col_width)
#
#     if inplace:
#         data.to_csv(dataset)

"""
import re
from keras.models import Model

def insert_layer_atindex(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='before'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name, 
                                                new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)
    
    logger("Adding the LSTM layer to the neural network")
    return Model(inputs=model.inputs, outputs=model_outputs)

def insert_layer(model,layer_name,layer_index):
    model = insert_layer_atindex(model, '.*activation.*', tf.keras.layers.Dropout(rate = ))
    
"""



# def generate_text(self, instruction, prefix=None, tuning=False,
#                   max_length=512,
#                   top_k=50,
#                   top_p=0.9,
#                   return_sequences=2,
#                   gpu=False,
#                   batch_size=32,
#                   save_path=os.getcwd()):
#     '''
#     Takes in initial text and generates text with specified number of characters more using Top P sampling
#     :param several parameters to hyperparemeterize with given defaults
#     :return: complete generated text
#     '''
#
#     if return_sequences < 1:
#         raise Exception("return sequences number is less than 1 (need an integer of atleast 1)")
#
#     if batch_size < 1:
#         raise Exception("Batch size must be equal to or greater than 1")
#
#     if max_length < 1:
#         raise Exception("Max text length must be equal to or greater than 1")
#
#     if not os.path.exists(save_path):
#         raise Exception("Save path does not exists")
#
#     if gpu:
#         if tf.test.gpu_device_name():
#             print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
#         else:
#             raise Exception("Please install GPU version of Tensorflow")
#
#         device = '/device:GPU:0'
#     else:
#         device = '/device:CPU:0'
#
#     sess = gpt2.start_tf_sess()
#
#     if tuning:
#         logger("Generating text from tuned model now...")
#         gpt2.load_gpt2(sess, run_name='run1')
#         gen_text = gpt2.generate(sess,
#                                  length=max_length,
#                                  temperature=0.7,
#                                  batch_size=batch_size,
#                                  prefix=prefix,
#                                  checkpoint_dir=save_path,
#                                  top_k=top_k,
#                                  top_p=top_p,
#                                  run_name='run1')
#         logger("Text Generation Complete")
#         logger("Storing information in client object under key 'Generated Text'")
#         self.models['Generated Text'] = gen_text
#         return self.models['Generated Text']
#     else:
#         model_name = "774M"
#         gpt2.download_gpt2(model_name=model_name)
#         gpt2.load_gpt2(sess, model_name=model_name)
#         logger("Generating text from pretrained model now")
#         gen_text = gpt2.generate(sess,
#                                  model_name=model_name,
#                                  prefix=prefix,
#                                  length=max_length,
#                                  temperature=0.7,
#                                  top_p=top_p,
#                                  nsamples=return_sequences,
#                                  batch_size=batch_size,
#                                  return_as_list=True
#                                  )
#         logger("Text Generation Complete")
#         logger("Storing information in client object under key 'Generated Text'")
#         self.models['Generated Text'] = gen_text
#         return self.models['Generated Text']
#
#
# def text_generation_query(self,
#                           save_path=os.getcwd(),
#                           batch_size=32,
#                           learning_rate=1e-4,
#                           save_every=500,
#                           steps=1000):
#     if save_every > steps or steps % save_every != 0:
#         raise Exception(
#             "You must have a save every value that is less than the number of steps (default 1000) and be a factor of steps")
#
#     data = DataReader(self.dataset)
#     data = data.data_generator()
#     gpt2.download_gpt2(model_name="124M")
#     sess = gpt2.start_tf_sess()
#     logger("Fine tuning the model now...")
#     gpt2.finetune(sess,
#                   dataset=data,
#                   model_name='124M',
#                   steps=steps,
#                   batch_size=batch_size,
#                   learning_rate=learning_rate,
#                   restore_from='fresh',
#                   run_name='run1',
#                   printevery=10,
#                   sample_every=200,
#                   save_every=save_every,
#                   checkpoint_dir=save_path
#                   )
#     logger("Finetuning complete - updated model saved at " + save_path + "run1")
#
