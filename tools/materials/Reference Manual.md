# Libra Reference Manual  

Libra is an open source deep learning API that provides a toolkit of intuitive and easily accessible machine learning methods, enabling users to streamline their data science workflows. The reference manual contains a detailed description of the Libra API. This reference delineates each of the methods avaiable for use, explaining how one can utilize them as well their respective parameters. The manual assumes that you have a baseline comprehension of some of the key concepts in machine learning. The reference is organized mainly in terms of location in the code where the method was defined as well as in accordance with OOP hierarchy, with a class introduced first and then its methods appearing underneath. Note that work on this manual is still in progress, and certain methods are due to recieve revised documentation in future updates.

***

Table of Contents
=================

* [dataset_labelmatcher.py](#dataset_labelmatcher)
   * [produceMask](#produceMask)
   * [get_similar_column](#get_similar_column)
   * [get_similar_model](#get_similar_model)
* [grammartree.py](#grammartree)
   * [get_value_instruction](#get_value_instruction)
* [pipeline_sample.py](#pipeline_sample)
   * [mypreprocessor](#mypreprocessor)
* [regression_split_functions.py](#regression_split_functions)
   * [initializer](#initializer)
   * [preprocesser](#preprocesser)
   * [instruction_identifier](#instruction_identifier)
   * [set_splitter](#set_splitter)
   * [modeler](#modeler)
   * [plotter](#plotter)
* [predictionModelCreation.py](#predictionModelCreation)
   * [get_keras_model_reg](#get_keras_model_reg)
   * [get_keras_model_class](#get_keras_model_class)
   * [get_keras_text_class](#get_keras_text_class)
   * [getKerasConvolutional](#getKerasConvolutional)
   * [load_model](#load_model)
* [tuner.py](#tuner)
   * [class CNNHyperModel](#class-CNNHyperModel)
      * [init](#__init__-CNNHyperModel)
      * [build](#build)
   * [tuneReg](#tuneReg)
   * [build_model](#build_model)
   * [tuneClass](#tuneClass)
   * [tuneCNN](#tuneCNN)  
* [generatePlots.py](#generatePlots)
  * [generate_clustering_plots](#generate_clustering_plots)
  * [generate_regression_plots](#generate_regression_plots)
  * [generate_classification_plots](#generate_classification_plots)
  * [generate_classification_together](#generate_classification_together)
  * [plot_loss](#plot_loss)
  * [plot_corr](#plot_corr)
  * [plot_acc](#plot_acc)
* [NLP_preprocessing.py](#NLP_preprocessing)
  * [get_target_values](#get_target_values)
  * [lemmatize_text](#lemmatize_text)
  * [tokenize_text](#tokenize_text)
  * [text_clean_up](#text_clean_up)
  * [fix_slang](#fix_slang)
* [data_preprocesser.py](#data_preprocesser)
  * [initial_preprocesser](#initial_preprocesser)
  * [structured_preprocesser](#structured_preprocesser)  
  * [process_dates](#process_dates)
  * [generate_column_labels](#generate_column_labels)
  * [clustering_preprocessing](#clustering_preprocessing)
  * [too_many_values](#too_many_values)
* [data_reader.py](#data_reader)
  * [class DataReader](#class-DataReader)
    * [init](#__init__-DataReader)
    * [retrieve_extension](#retrieve_extension)
    * [data_generator](#data_generator)
    * [get_available_gpus](#get_available_gpus)
    * [is_gpu_available](#is_gpu_available)
    * [random_trim](#random_trim)
    * [trim_gpu](#trim_gpu)
* [huggingfaceModelRetrainHelper.py](#huggingfaceModelRetrainHelper)
  * [train](#train)
  * [class CustomDataset](#class-CustomDataset)
    * [init](#__init__-CustomDataset)
    * [len](#__len__)
    * [getitem](#__getitem__)
  * [inference](#inference)
* [image_caption_helpers.py](#image_caption_helpers)
  * [get_path_column](#get_path_column)
  * [load_image](#load_image)
  * [map_func](#map_func)
  * [class BahdanauAttention](#class-BahdanauAttention)
    * [init](#__init__-BahdanauAttention)
    * [call](#call-BahdanauAttention)
  * [class CNN_Encoder](#class-CNN_Encoder)
    * [init](#__init__-CNN_Encoder)
    * [call](#call-CNN_Encoder)
  * [class CNN_Encoder](#class-CNN_Encoder)
    * [init](#__init__-RNN_Decoder)
    * [call](#call-RNN_Decoder)
    * [reset_state](#reset_state)
  * [generate_caption_helper](#generate_caption_helper)
* [image_preprocesser.py](#image_preprocesser)
  * [setwise_preprocessing](#setwise_preprocessing)
  * [pathwise_preprocessing](#pathwise_preprocessing)
  * [classwise_preprocessing](#classwise_preprocessing)
  * [process_class_folders](#process_class_folders)
  * [add_resized_images](#add_resized_images)
  * [replace_images](#replace_images)
  * [create_folder](#create_folder)
  * [save_image](#save_image)
  * [calculate_medians](#calculate_medians)
  * [process_color_channel](#process_color_channel)
* [classification_models.py](#classification_models)
* [dimensionality_red_queries.py](#dimensionality_red_queries)
  * [dimensionality_reduc](#dimensionality_reduc)
  * [dimensionality_RF](#dimensionality_RF)
  * [dimensionality_PCA](#dimensionality_PCA)
  * [dimensionality_ICA](#dimensionality_ICA)
  * [get_last_file](#get_last_file)
* [feedforward_nn.py](#feedforward_nn)
* [nlp_queries.py](#nlp_queries)
* [supplementaries.py](#supplementaries)
* [predictionQueries.py](#predictionQueries)
  * [clearLog](#clearLog)
  * [logger](#logger-predictionQueries)
  * [class client](#class-client)
    * [init](#__init__-client)
    * [get_models](#get_models)
    * [predict](#predict)
    * [neural_network_query](#neural_network_query)
    * [regression_query_ann](#regression_query_ann)
    * [classification_query_ann](#classification_query_ann)
    * [kmeans_clustering_query](#kmeans_clustering_query)
    * [svm_query](#svm_query)
    * [nearest_neighbor_query](#nearest_neighbor_query)
    * [decision_tree_query](#decision_tree_query)
    * [allClassQuery](#allClassQuery)
    * [tune](#tune)
    * [stat_analysis](#stat_analysis)
    * [convolutional_query](#convolutional_query)
    * [encode_text](#encode_text)
    * [predict_text_sentiment](#predict_text_sentiment)
    * [text_classification_query](#text_classification_query)
    * [get_summary](#get_summary)
    * [summarization_query](#summarization_query)
    * [dimensionality_reducer](#dimensionality_reducer)
    * [show_plots](#show_plots)
    * [model_data](#model_data)
    * [operators](#operators)
    * [accuracy](#accuracy)
    * [losses](#losses)

***

## dataset_labelmatcher ##

### produceMask ###

``` python
libra.produceMask(instruction_label)
```

Generates/stores with base object an encoding mask of form list ```[int/float, int/float, int/float]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


*Parameters --*

instruction_label: `str`

String based of the instruction sent to encode in the form of a mask


*Returns --*

mask: `[int/float, int/float, int/float]`

Retrieves a list of encoded numerical representation of the instruction text recieved

### get_similar_column ###

``` python
libra.get_similar_column(instruction, dataset)
```

Identifies/saves the column in dataset whose title shares the most similarity with keyword in instruction sent in written query 


*Parameters --*

instruction: `str`

String representation of the instruction sent in written query

dataset

Data to be analyzed that is selected/sent via written query (is by default set to dataframe format)


*Returns --*

dataset.columns[idx]

Retrieves the column whose index shares closest similarity via comparision of their Levennshtein distance to elements in instruction string  

### get_similar_model ###

``` python
libra.get_similar_model(model_requested, model_keys)
```

Identifies/saves the model requested in instruction whose title shares the most similarity with models available in Libra 


*Parameters --* 

model_requested: `str`

String representation of name of model parsed from instruction sent in written query

model_keys: `[str, str,..., str]`

List of string "key" representations of models inbuilt/supported in Libra for use 


*Returns --*

model.keys[idx]: `str`

Retrieves the model whose index shares closest similarity via comparision of their Levennshtein distance to that of the model name requested in the instruction string

***

## grammartree ##

### get_value_instruction ###

``` python
libra.get_value_instruction(sent)
```

Acquires/grammatically parses/reformats (using Textblob) relevant parts of speech for analysis from instruction sent to/in written query into instruction interpretable by other Libra methods

(Note: If an issue ensues in the instruction where the method cannot comprehend the instruction, an error message will appear stating to 'Please try re-typing your sentence') 


*Parameters --*

sent: `str`

String representation of the instruction sent via written query


*Returns --*

decoded: `str` 

Retrieves specific strings from written instruction to rearranged instruction usable for subsequent methods

***

## pipeline_sample ##

### mypreprocessor ###

``` python
libra.mypreprocessor(model_requested, model_keys)
```

(For dev use only) Allows for testing of customized feature of preprocessing of dataset 


*Parameters --*

instruction: `str`

String representation of the instruction sent in written query

path_to_set: `str`

String representation of name of path location where data is located


*Returns --*

None

Implements preprocessing algorithm on basis of defined parameters

***


## regression_split_functions ##

### initializer ###

``` python
libra.initializer(params)
```

Initializes the data pipeline with given set of data for backend regression based modelling tasks 

*Parameters --*

params: `{}`

Dictionary of parameters for regression task data pipeline 


*Returns --*

params: `{}`

Retrieves dictionary of parameters revised after reading in data, removing data of any incomplete observations, and initializing the data parameter with inputted set of data
   
### preprocesser ###
   
``` python
libra.preprocesser(params)
```

Implements preprocessing code defined in structured_preprocessor method to data earlier initialized in data pipeline parameters dictionary for regression based tasks


*Parameters --* 

params: `{}`

Dictionary of parameters for regression task data pipeline


*Returns --* 

None

Executes structured_preprocesser method which preprocesses data by categorizing/labelling/transforming data into format computable for regression model

### instruction_identifier ###

``` python
libra.instruction_identifier(params)
```

Employs usage of both get_similar_colummn and get_value_instruction methods to identify appropriate column and specific regression task type for analysis using initialized data pipeline parameters dictionary 


*Parameters --*

params: `{}`

Dictionary of parameters for regression task data pipeline


*Returns --*

None

Executes both get_similar_column and get_value_instruction methods to determine column and task type for regression analysis and updates y parameter in initialized data pipeline parameters dictionary

### set_splitter ###

``` python
libra.set_splitter(params)
```

Generates/stores training/testing segments of data for both class and target variable in initialized data pipeline parameters dictionary


*Parameters --*

params: `{}`

Dictionary of parameters for regression task data pipeline 


*Returns --*

None

Executes training testing splitting method based on scikit-learn functionality to construct/store seperate training and testing dataset in dictionary of parameters for regression task data pipeline
   
### modeler ###

``` python
libra.modeler(params)
```

Develops/stores regression neural network model for analysis through application of get_keras_model_reg method as a parameter in regression data pipeline parameters dictionary 


*Parameters --* 

params: `{}`

Dictionary of parameters for regression task data pipeline


*Returns --*

None 

Executes get_keras_model_reg method to develop regression neural network that is fitted and rerun with number of layers until validation loss is maximized with history of iteration of models and associated validation loss metrics assigned to models and losses parameters of the regression data pipeline parameters dictionary
   
### plotter ###

``` python
libra.plotter(params)
```

Generates all plots affiliated with regression neural network model using generateRegressionPlots method and stores them in initialized data pipeline parameters dictionary


*Parameters --* 

params: `{}`

Dictionary of parameters for regression task data pipeline


*Returns --*

None

Executes generateRegressionPlots method to create all plots displaying the state of the various previous model iterations that is assigned to the plots parameter of the regression data pipeline parameters dictionary

***

## predictionModelCreation ##

### get_keras_model_reg ###

``` python
libra.get_keras_model_reg(dataset, i)
```

Constructs a regression neural network that uses rectified linear unit activation function and a normal distribution for its initial weight


*Parameters --* 

dataset

Data to be analyzed that is selected/sent via written query (is by default set to dataframe format)

i: `int`

Number of additional layers used after creation of baseline regression model


*Returns --*

model: `obj` 

Retrieves final regression neural network model with a rectified linear unit activation function and normal kernel initializer compiled using a mean square error loss function and adaptive first/second moment stochastic gradient descent optimizer

### get_keras_model_class ###

``` python
libra.get_keras_model_class(dataset, i, num_classes)
```

Constructs a classification neural network that uses rectified linear unit activation function and a normal distribution for its initial weight


*Parameters --* 

dataset

Data to be analyzed that is selected/sent via written query (is by default set to dataframe format)

i: `int`

Integer number of additional layers used after creation of baseline classification model

num_classes: `int`

Integer number of classes associated with data for classification model


*Returns --*

model: `obj`

Retrieves final classification neural network model with a rectified linear unit activation function and normal kernel initializer with softmax activation function outer layer compiled using a categorical cross-entropy loss function and adaptive first/second moment stochastic gradient descent optimizer as well as including accuracy metrics
   
### get_keras_text_class ###

``` python
libra.get_keras_text_class(max_features, num_classes)
```

Constructs a long term short memory recurrent binary text classification neural network that uses softmax activation function


*Parameters --* 

max_features: `int`

Integer maximum number of feature variables (measurable properties found in other column segments of data) to take into account in text classification model

num_classes: `int`

Integer number of classes associated with data for text classification model


*Returns --*

model: `obj`

Retrieves final text classification neural network model with a softmax activation function outer layer compiled using a binary cross-entropy loss function and adaptive first/second moment stochastic gradient descent optimizer as well as including relevant accuracy metrics
   
### getKerasConvolutional ###

``` python
libra.instruction_getKerasConvolutional()
```

Updates user of successful entry of instruction to request convolutional neural network from written query


*Parameters --*

None


*Returns --*

None

Executes method to send message to user that indicates convolutional neural network request has been recorded

### load_model ###

``` python
libra.load_model(self, load_path)
```

Loads a user defined model from user specified model file path if the user has provided the model dictionary


*Parameters --* 

load_path: `str`

String representation of file path name to load pretrained user model from location


*Returns --*

loaded_model: `obj`

Retrieves model with initialized weights that was stored in json formatted file from specified path

***

## tuner ##

### class CNNHyperModel ###

``` python
class CNNHyperModel(HyperModel)
```

A class whose Hypermodel object instance enables the hyperparameter tuning of an inputted neural network. The list of its methods are provided below.

#### __init__-CNNHyperModel ####

``` python
CNNHyperModel.__init__(self, input_shape, num_classes)
```

Initializes the hyperparameter model class for neural network tuning 


*Parameters --*

input_shape: `numpy.array`

Array of dimensional shape needed to fit data into the neural network model (is assumed to be of a 4-D variety)

num_classes: `int`

Integer number of classes associated with data for convolutional neural network hyperparameter model


*Returns --*

None

Executes code to initialize neural network hypermodel with parameters of an initial input structure and number of classes for analysis

#### build ####

``` python
CNNHyperModel.build(self, hp)
```

Generates/stores different layers of neural network model with a selection of parameters which the model selects to use during runtime that are optimal


*Parameters --*

hp: `{}`

Dictionary of hyperparameter options 


*Returns --* 

model: `obj`

Retrieves neural network hypermodel with numerous inbuilt options per parameter enabled (activation function choices can be rectified linear unit/tahn/sigmoid based, etc.) to choose upon startup, categorical cross-entropy loss function, logarithmic based sampling, adaptive first/second moment stochastic gradient descent optimization, and relevant accuracy metrics

### tuneReg ###

``` python
libra.tuneReg(data, 
              target, 
              max_layers=10, 
              min_layers=2, 
              min_dense=2, 
              max_dense=512, 
              executions_per_trial=3,
              max_trials=1)
```

Performs automatic tuning of prior defined hyperparameters varying across range of layers and number of units in dense layers across selected trials for finding optimal regression neural network model


*Parameters --*

data

Data selected and sent via written query to instance by user to be analyzed 

target

Target variable of analysis the model seeks to optimize (also known as hyperparameter metric)

max_layers=10 (`int`)

Integer maximum number of layers composed in model (is set by default to 10)

min_layers=2 (`int`)

Integer minimum number of layers composed in model (is set by default to 2)

min_dense=2 (`int`)

Integer minimum number of units in a dense layer of model (is set by default to 2)

max_dense=512 (`int`)

Integer maximum number of units in a dense layer of model (is set by default to 512)

executions_per_trial=3 (`int`)

Integer number of models that are built and fit per trial to ensure robustness of models (is set by default to 3)

max_trials=1 (`int`)

Integer number of hyperparameter combinations that are tested by the tuner (is set by default to 1)


*Returns --* 

models[0]: `obj`

Retrieves best model by searching in hyperparameter defined tuner space from generated scenario models used while tuning hyperparameters of regression neural network model

### build_model ###

``` python
libra.build_model(hp)
```

Constructs regression/convolution neural networks based off identified hyperparameters 


*Parameters --*

hp: `{}`

Dictionary of hyperparameter options 


*Returns --* 

model: `obj`

Retrieves baseline neural network model optimized using adaptive first/second moment stochastic gradient descent method with outer dense softmax layer 

### tuneClass ###

``` python
libra.tuneClass(X, 
                y, 
                num_classes,
                max_layers=10,
                min_layers=2,
                min_dense=32,
                max_dense=512,
                executions_per_trial=3,
                max_trials=1,
                activation='relu',
                loss='categorical_crossentropy',
                metrics='accuracy')
```

Performs automatic tuning of prior defined hyperparameters varying across range of layers and number of units in dense layers amongst selected trials for finding optimal classification neural network


*Parameters --*

X

Variable/vector of input features in data for classification neural network model analysis

y

Prediction variable/vector of classes in data for classification neural network model analysis 

data

Data selected and sent via written query to instance by user to be analyzed 

target

Target variable of analysis the model seeks to optimize (also known as hyperparameter metric)

max_layers=10 (`int`)

Integer maximum number of layers composed in model (is set by default to 10)

min_layers=2 (`int`)

Integer minimum number of layers composed in model (is set by default to 2)

min_dense=2 (`int`)

Integer minimum number of units in a dense layer of model (is set by default to 2)

max_dense=512 (`int`)

Integer maximum number of units in a dense layer of model (is set by default to 512)

executions_per_trial=3 (`int`)

Integer number of models that are built and fit per trial to ensure robustness of models (is set by default to 3)

max_trials=1 (`int`)

Integer number of hyperparameter combinations that are tested by the tuner (is set by default to 1)

activation='relu' (`str`)

String representation of the type of activation function used by the model (is set by default to 'relu', or rectified linear unit based activation function)

loss='categorical_crossentropy' (`str`)

String representation of the type of loss function used by the model (is set by default to 'categorical_crossentropy', or categorical cross-entropy loss function that measures model based of logarithmic loss)

metrics='accuracy' (`str`)

String representation of the type of predictive metric used to assess the performance of the model (is set by default to 'accuracy', or the accuracy classification score)


*Returns --* 

models[0]: `obj`

Retrieves best model by searching in hyperparameter defined tuner space from generated scenario models used while tuning hyperparameters of classification neural network model

### tuneCNN ###  

``` python
libra.tuneCNN(X, y, num_classes)
```

Performs automatic tuning of prior defined hyperparameters to find optimal convolutional neural network model


*Parameters --*

X

Variable/vector of input features in data for classification neural network model analysis

y

Prediction variable/vector of classes in data for classification neural network model analysis 

num_classes: `int`

Integer number of classes associated with data for classification model


Returns -- 

`tuner.get_best_models(1)[0]`: `obj` 

Retrieves best model by searching in hyperparameter defined tuner space from generated scenario models used while tuning hyperparameters of convolutional neural network model

***

## generatePlots ##

### generate_clustering_plots ###

``` python
libra.generate_clustering_plots(kmeans, dataPandas, dataset)
```

Generates/stores all relevant plots for clustering models


*Parameters --*

Retrieves list containing all the saved generated plotskmeans

Options specifying the type of kmeans clustering technique used by the model 

dataPandas

Name of inbuilt Pandas dataframe acquiring data from dataset specifically used to store x and y axes for plotting

dataset

Data to be analyzed that is selected/sent via written query (is by default set to dataframe format)


*Returns --* 

plots: `[]`

Retrieves list containing all the saved generated plots

plot_names: `[str, str,..., str]`

Retrieves list of string representation or label names for plots generated 

### generate_regression_plots ###

``` python
libra.generate_regression_plots(history, data, label)
```

Generates/stores all relevant plots for regression models


*Parameters --*

history

History of past model iterations/results from regression model being plotted

data

Data selected and sent via written query to instance by user to be analyzed (is set by default to have dataframe format)

label: `[str, str]`

String representation of labels used for names for x/y axes of plots generated


plots: `[]`

Retrieves list containing all the saved generated plots

plot_names: `[str, str,..., str]`

Retrieves list of string representation or label names for plots generated 

### generate_classification_plots ###

``` python
libra.generate_classification_plots(history, data, label, model, X_test, y_test)
```

Generates/stores all relevant plots for classification models


*Parameters --*

history

History of past model iterations/results from classification model being plotted

data

Data selected and sent via written query to instance by user to be analyzed (is set by default to have dataframe format)

label: `[str, str]`

String representation of labels used for names for x/y axes of plots generated

model: `obj`

Classification model object being plotted

X_test

Test variable/vector of input features in data for classification model analysis

y_test

Test prediction variable/vector of classes in data for classification model analysis 


*Returns --* 

return_plots: `{}`

Retrieves dictionary of dynamically fitting compilation of plots with key of plot names

### generate_classification_together ###

``` python
libra.generate_regression_plots(history, data, model, X_test, y_test)
```

Generates/stores validation loss and accuracy score plots for classification models side-by-side in same pane


*Parameters --*

history

History of past model iterations/results from classification model being plotted

data

Data selected and sent via written query to instance by user to be analyzed (is set by default to have dataframe format)

label: `[str, str]`

String representation of labels used for names for x/y axes of plots generated

model: `obj`

Classification model object being plotted

X_test

Test variable/vector of input features in data for classification model analysis

y_test

Test prediction variable/vector of classes in data for classification model analysis 


*Returns --* 

None

Executes code to output plot pane where graph depicting loss validation appears above and graph depicting score accuracy appears below

### plot_loss ###

``` python
libra.plot_loss(history)
```

Generates/stores validation loss plot for any given machine learning technique model


*Parameters --* 

history

History of past model iterations/results from classification model being plotted


*Returns -- *

img: `Figure` 

Retrieves model loss figure which has x/y axes of validation loss/epoch measured across both training and testing datasets

### plot_corr ###


``` python
libra.plot_corr(data, col=[])
```

Generates/stores correlation plot for any columns available in dataset for analysis


*Parameters --*

data

Data selected and sent via written query to instance by user to be analyzed (is set by default to have dataframe format)

col=`[]` (`[str, str,..., str]`)

List of string representation of names of columns where correlation is being sougth (is set by default to an empty plot, which assumes creation of plot checking correlation between all variables)


*Returns --*

img: `Figure` 

Retrieves square correlation heatmap figure shaded in accordance with strength of correlation coefficient relating two variables

### plot_acc ###

``` python
libra.plot_acc(history)
```

Generates/stores accuracy plot for any given machine learning technique model


*Parameters --* 

history

History of past model iterations/results from classification model being plotted


*Returns -- *

img: `Figure` 

Retrieves model accuracy figure which has x/y axes of accuracy score/epoch measured across both training and testing datasets

***

## NLP_preprocessing ##

### get_target_values ###

``` python
libra.get_target_values(data, instruction, yLabel)
```

### lemmatize_text ###

``` python
libra.lemmatize_text(dataset)
```

### tokenize_text ###

``` python
libra.tokenize_text(dataset)
```

### text_clean_up ###

``` python
libra.text_clean_up(dataset)
```

### fix_slang ###

``` python
libra.fix_slang(text)
```

***

## data_preprocesser ##

### initial_preprocesser ###

``` python
libra.initial_preprocesser(data, instruction, preprocess)
```

### structured_preprocesser ###

``` python
libra.structured_preprocesser(data)
```

### process_dates ###

``` python
libra.process_dates(data)
```

### generate_column_labels ###

``` python
libra.generate_column_labels(pipeline, numeric_cols)
```

### clustering_preprocessor ###

``` python
libra.clustering_preprocessor(data)
```

### too_many_values ###

```python
libra.too_many_values(data, mca_threshold)
```

***

## data_reader ##

### class-DataReader ###

``` python
class DataReader()
```
    
#### __init__-DataReader ####

``` python
DataReader.__init__(self, filepath)
```

#### retrieve_extension ####

``` python
DataReader.retrieve_extension(self)
```
    
#### data_generator ####

``` python
DataReader.data_generator(self)
```

#### get_available_gpus ####

``` python
DataReader.get_available_gpus(self)
```

#### is_gpu_available ####

``` python
DataReader.is_gpu_available(self)
```

#### random_trim ####

``` python
DataReader.random_trim(self)
```

#### trim_gpu ####

``` python
DataReader.trim_gpu(self)
```

***

## huggingfaceModelRetrainHelper ##

### train ###

### class-CustomDataset ###

#### __init__-CustomDataset ####

#### __len__ #### 

#### __getitem__ ####

### inference ###

***


## image_caption_helpers ##

### get_path_column ###

```python
libra.get_path_column(df)
```

### load_image ###

```python
libra.load_image(image_path)
```

### map_func ###

```python
libra.map_func(img_name, cap)
```

### class-BahdanauAttention ###

``` python
class BahdanauAttention()
```
    
#### __init__-BahdanauAttention ####

``` python
BahdanauAttention.__init__(self, units)
```

#### call-BahdanauAttention ####

``` python
BahdanauAttention.call(self, features, hidden)
```
    
### class-CNN_Encoder ###

``` python
class CNN_Encoder()
```

#### __init__-CNN_Encoder ####

``` python
CNN_Encoder.__init__(self, embedding_dim)
```

#### call-CNN_Encoder ####

``` python
CNN_Encoder.call(self)
```

### RNN_Decoder ###

``` python
class RNN_Decoder()
```

#### __init__-RNN_Encoder ####

``` python
RNN_Decoder.__init__(self, embedding_dim, units, vocab_size)
```

#### call-RNN_Encoder ####

``` python
RNN_Decoder.call(self, x, features, hidden)
```

#### reset_state ####

``` python
RNN_Decoder.reset_state(self, batch_size)
```

### generate_caption_helper ###

```python
libra.generate_caption_helper(image, decoder, encoder, tokenizer, image_features_extract_model, max_length=500)
```

***

## image_preprocesser ##

### setwise_preprocessing ###

``` python
libra.setwise_preprocessing(data_path, new_folder=True)
```

### setwise_preprocessing ###

``` python
libra.pathwise_preprocessing(csv_file, dath_paths, label, image_column, training_ratio)
```

### setwise_preprocessing ###

``` python
libra.classwise_preprocessing(data_path, training_ratio)
```

### process_class_folders ###

``` python
libra.process_class_folder(data_path)
```

### replace_images ###

``` python
libra.replace_images(data_path, loaded_shape)
```

### create_folder ###

``` python
libra.create_folder(path, folder_name)
```

### save_image ###

``` python
libra.save_image(path, img, img_name, classification)
```

### calculate_medians ###

``` python
libra.calculate_medians(heights, widths)
```

### process_color_channel ###

``` python
libra.process_color_channel(img, height, width)
```

***

## classification_models ##

***

## dimensionality_red_queries ##
 
### dimensionality_reduc ###

``` python
libra.dimensionality_reduc(instruction, dataset, arr = ["RF", "PCA", "ICA"], inplace = True)
```

### dimensionality_RF ###

``` python
libra.dimensionality_RF(instruction, dataset, target="", y="", n_features=10)
```

### dimensionality_PCA ###

``` python
libra.dimensionality_PCA(instruction, dataset, target="", y="", n_components=10)
```

### dimensionality_ICA ###

``` python
libra.dimensionality_ICA(instruction, dataset, target="", y="")
```

### get_last_file ###

``` python
libra.get_last_file()
```

***

## feedforward_nn ##

***

## nlp_queries ##

***

## supplementaries ##

***

## predictionQueries ##

### clearLog ###

``` python
libra.clearLog()
```

Clears the logger hierarchial display of any past record of processes used in Libra session


*Parameters --*

None


*Returns --*

None

Instantiates/reassigns the global counter and logger state variable back to 0, refreshing contents of logger

***

### logger-predictionQueries ###

``` python
libra.logger(instruction, found="", slash='')
```

Generates hierarchial display and tracks record of processes in operation upon utilizing a particular method in Libra


*Parameters --*

instruction: `str`

String representation of request from written query from methods which can utilize the logger

found: "" (`str`)

String representation of prior processes initiated upon initializing class instance or application (is set by default to ""/blank)

slash: "" (`str`)

String indicator used to structure/organize logger output


*Returns --*

None

Executes/updates hierarchial display upon usage of supported method such as a written query

***

### class-client ###

``` python
class client()
```

A class whose instance enables a "client session" where written query methods allow users to apply machine learning techniques for analysis directly to data problems. The list of its methods are provided below.

#### __init__-client ####

``` python
client.__init__(self, data)
```

Initializes the client class instance 


*Parameters --*

data

Data selected and sent via written query by user to be analyzed by methods of the client class


*Returns*

None

Executes the creation of the client class instance in the form of creating the client class object, loading/formatting the assigned data, restoring old models, storing new models, and clearing the log upon exit of class instance


#### get_models ####

``` python
client.get_models(self, model_requested)
```

Acquires previously defined model based off request from written query sent to/in client class instance


*Parameters --*

model_requested: `str`

String "key" representation of model which was intuited from the instruction found in the written query 


*Returns --* 

get_similar_model(model_requested, self.models.keys())

Retrieves the model by retrieving the model string key matching one of the supported options in the list of model keys made accessible to the client class instance

#### predict ####

``` python
client.predict(self, modelKey, data)
```

Generates/stores list of prediction results from model identified as requested from written query sent to/in the client class instance

*Parameters --*

modelKey: `str`

String "key" representation of model that needs to be utilized to make prediction intuited from written query 

data

Data of the desired prediction set that is intialzed in the client class instance (is set by default to be in dataframe format)


*Returns --*

predictions: `[str, str,..., str] or [float, float,..., float]` 

Retrieves a list of the string or predictions for model associated with key string value depending on activation of interpreter itself dependent on nature of data initialized in client class instance

#### neural_network_query ####

``` python
client.neural_network_query(self, 
                            instruction,
                            drop=None,
                            preprocess=True,
                            test_size=0.2,
                            random_state=49,
                            epochs=50,
                            generate_plots=True,
                            callback_mode='min',
                            maximizer="val_loss",
                            save_model=True,
                            save_path=os.getcwd())
```


Generates artificial neural network model that choose between classification or regression problem request from written query sent to/in client class instance 


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

preprocess: True (`bool`)

Option to preprocess the data used by the model (is set by default to true)

callback_mode: "min" (`str`)

String representing type of saving mechanism for multiple epochs of training of model (is set by default to min/epoch iteration of model will be saved only if maximizer function gets increased or in this case the val_loss function is reduced)

drop: None (`str`)

Option to drop any columns during analysis of data (is set by default is set to None)

random_state: 49 (`int`)

Random value assigned that ensures the shuffling of the dataset before applying the test/training split remains the same across multiple runs (is set by default to 49)

test_size: 0.2 (`float`)

Percentage measure of the proportion of the data to include in test split of the dataset (is set by default to 0.2)

epochs: 5 (`int`)

Number of full cycles used for model fitting/training on the training dataset (is set by default to 5)

generate_plots: True (`bool`)

Option to generate all relevant plots for query (is set by default to true)

maximizer: "val_loss" (`str`)
 
String representing type of model optimization function to use (is set by default to val_loss)

save_model: True (`bool`)

Option to store the iterations of the model in memory (is set by default to true)

save_path: os.getcwd() (`None`)

File path in computer directory where artificial neural network model is saved in (is set by default to os.getcwd()/current working directory)


*Returns --* 

None

Executes artificial neural network model training/testing and outputs resulting summary/metrics/associated plots

#### regression_query_ann ####

``` python
client.regression_query_ann(self, 
                            instruction,
                            drop=None,
                            preprocess=True,
                            test_size=0.2,
                            random_state=49,
                            epochs=50,
                            generate_plots=True,
                            callback_mode='min',
                            maximizer="val_loss",
                              save_path=os.getcwd())
```

Generates feed-forward artificial neural network model used to solve only regression problem request from written query sent to/in client class instance 


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

preprocess: True (`bool`)

Option to preprocess the data used by the model (is set by default to true)

callback_mode: "min" (`str`)

String representing type of saving mechanism for multiple epochs of training of model (is set by default to min/epoch iteration of model will be saved only if maximizer function gets increased or in this case the val_loss function is reduced)

drop: None (`str`)

Option to drop any columns during analysis of data (is set by default is set to None)

random_state: 49 (`int`)

Random value assigned that ensures the shuffling of the dataset before applying the test/training split remains the same across multiple runs (is set by default to 49)

test_size: 0.2 (`float`)

Percentage measure of the proportion of the data to include in test split of the dataset (is set by default to 0.2)

epochs: 5 (`int`)

Number of full cycles used for model fitting/training on the training dataset (is set by default to 5)

generate_plots: True (`bool`)

Option to generate all relevant plots for query (is set by default to true)

maximizer: "val_loss" (`str`)
 
String representing type of model optimization function to use (is set by default to val_loss)

save_model: True (`bool`)

Option to store the iterations of the model in memory (is set by default to true)

save_path: os.getcwd() (`None`)

File path in computer directory where artificial neural network model is saved in (is set by default to os.getcwd()/current working directory)


*Returns --* 

None

Executes regression feed-forward artificial neural network model training/testing and outputs resulting summary/metrics/associated plots

#### classification_query_ann ####

``` python
client.classification_query_ann(self, 
                                instruction,
                                preprocess=True,
                                callback_mode='min',
                                drop=None,
                                random_state=49,
                                test_size=0.2,
                                epochs=5,
                                generate_plots=True,
                                maximizer="val_loss",
                                save_model=True,
                                save_path=os.getcwd())
```

Generates feed-forward artificial neural network model used to solve only multilabel classification problem request from written query sent to/in client class instance 


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

preprocess: True (`bool`)

Option to preprocess the data used by the model (is set by default to true)

callback_mode: "min" (`str`)

String representing type of saving mechanism for multiple epochs of training of model (is set by default to min/epoch iteration of model will be saved only if maximizer function gets increased or in this case the val_loss function is reduced)

drop: None (`str`)

Option to drop any columns during analysis of data (is set by default is set to None)

random_state: 49 (`int`)

Random value assigned that ensures the shuffling of the dataset before applying the test/training split remains the same across multiple runs (is set by default to 49)

test_size: 0.2 (`float`)

Percentage measure of the proportion of the data to include in test split of the dataset (is set by default to 0.2)

epochs: 5 (`int`)

Number of full cycles used for model fitting/training on the training dataset (is set by default to 5)

generate_plots: True (`bool`)

Option to generate all relevant plots for query (is set by default to true)

maximizer: "val_loss" (`str`)
 
String representing type of model optimization function to use (is set by default to val_loss)

save_model: True (`bool`)

Option to store the iterations of the model in memory (is set by default to true)

save_path: os.getcwd() (`None`)

File path in computer directory where artificial neural network model is saved in (is set by default to os.getcwd()/current working directory)


*Returns --* 

None

Executes classification feed-forward artificial neural network model training/testing and outputs resulting summary/metrics/associated plots

#### kmeans_clustering_query ####

``` python
client.kmeans_clustering_query(self, 
                               instruction,
                               preprocess=True,
                               generate_plots=True,
                               drop=None,
                               base_clusters=1)
```

Generates kmeans clustering model used to solve cluster analysis problem request from written query sent to/in client class instance 


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

preprocess: True (`bool`)

Option to preprocess the data used by the model (is set by default to true)

generate_plots: True (`bool`)

Option to generate all relevant plots for query (is set by default to true)

drop: None (`str`)

Option to drop any columns during analysis of data (is set by default is set to None)

base_clusters: 1 (`int`)

Value specifying the number of clusters the kmeans clustering algorthim uses as a baseline when developing the model (is set by default to 1)


*Returns --* 

None

Executes k means clustering model training/testing and outputs resulting summary/metrics/associated plots

#### svm_query ####

``` python
client.svm_query(self, 
                 instruction,
                 test_size=0.2,
                 kernel="linear",
                 preprocess=True,
,                drop=True,
                 cross_val_size=0.3)
```

Generates support vector machine model used to solve classification problem request from written query sent to/in client class instance 


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

test_size: 0.2 (`float`)

Percentage measure of the proportion of the data to include in test split of the dataset (is set by default to 0.2)

kernel: "linear" (`str`)

String representing type of kernel function used by the support vector machine model

preprocess: True (`bool`)

Option to preprocess the data used by the model (is set by default to true)

drop: True (`bool`)

Option to drop any columns during analysis of data (default is set to None)

cross_val_size: 0.3 (`float`)

Percentage measure of the proportion of the data to include in test split of the dataset for cross validation metrics (is set by default to 0.3)


*Returns --* 

svm: `object` 

Retrieves the support vector machine classification model after training/testing and storing its information in client instance class
    
#### nearest_neighbor_query ####


``` python
client.nearest_neighbor_query(self, 
                              instruction,
                              preprocess=True,
                              drop=None,
                              min_neighbors=3,
                              max_neighbors=10)
```

Generates k nearest neighbor model used to solve classification problem request from written query sent to/in client class instance 


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

preprocess: True (`bool`)

Option to preprocess the data used by the model (is set by default to true)

drop: None (`str`)

Option to drop any columns during analysis of data (default is set to None)

min_neighbors: 3 (`int`)

Minimum number of closest neighboring training examples in the feature space to a particular observation in dataset initialized for analysis in client class instance (default is set to 3)

max_neighbors: 10 (`int`)

Maximum number of closest neighboring training examples in the feature space to a particular observation in dataset initialized for analysis in client class instance (default is set to 10)


*Returns --* 

knn: `object` 

Retrieves the k nearest neighbor classification model after training/testing and storing its information in client instance class

#### decision_tree_query ####

``` python
client.decision_tree_query(self, 
                           instruction,
                           preprocess=True,
                           test_size=0.2,
                           drop=None)
```

Generates decision tree model used to solve classification problem request from written query sent to/in client class instance 


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

preprocess: True (`bool`)

Option to preprocess the data (is set by default to true)

test_size: 0.2 (`float`)

Percentage measure of the proportion of the data to include in test split of the dataset (is set by default to 0.2)

drop: None (`str`)

Option to drop any columns during analysis of data (default is set to None)


*Returns --* 

clf: `object` 

Retrieves the classification model after training/testing and storing its information in client instance class

#### allClassQuery ####

``` python
client.allClassQuery(self,
                     instruction,
                     preprocess=True,
                     test_size=0.2,
                     drop=None,
                     random_state=49,
                     save_model=1)
```

Generates all traditional machine learning classification models (decision tree/nearest neighbor/support vector machine)/selects best one used to solve classification problem request from written query sent to/in client class instance 


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

preprocess: True (`bool`)

Option to preprocess the data (is set by default to true)

test_size: 0.2 (`float`)

Percentage measure of the proportion of the data to include in test split of the dataset (is set by default to 0.2)

drop: None (`str`)

Option to drop any columns during analysis of data (default is set to None)

random_state: 49 (`int`)

Random value assigned that ensures the shuffling of the dataset before applying the test/training split remains the same across multiple runs (is set by default to 49)

save_model: 1 (`int`)

Option to specify whether to save model (is set by default to 1 which is to save the model in the format of a Tensorflow SavedModel file)


*Returns --*

models[scores.index(max(scores))]: 'object'

Retrieves the classification model with the highest metric scores on data after training/testing each model and storing their information in client instance class

#### tune ####

``` python
client.tune(self, model_to_tune)
```

Optimizes hyperparameters controlling the learning process of a neural network model depending on the type of that model previously requested from written query sent to/in client class instance


*Parameters --*

model_to_tune: `object`

Neural network model previously instantiated in client class instance 


*Returns --*

returned_model: `object`

Neural network model that is tuned from the prior variant of the selected model

#### stat_analysis ####

``` python
client.stat_analysis(self, column_name="none", drop=None)
```

Generates formatted tabular dataset providing synthesis of key statistics/features of given data 


*Parameters --*

column_name: none (`str`)

String that is the column name for column the user chooses to analyze from dataset initalized in client class instance (default is set to none/assumption is that all columns will be subject for analysis)

drop: None

Option to drop any columns from analysis of dataset initialized for analysis in client class instance (default is set to none) 


*Returns --*

None

Executes all encoding/processing/similarity calculations to outputs resulting tabulated statistical description/feature selection of dataset initialized for analysis in client class instance

#### convolutional_query ####

``` python
client.convolutional_query(self, data_path=None, new_folders=True)
```

Generates/stores convolutional neural network model (dynamically utilizing images acquired from automated web browsing session) used to solve visual analysis problem request from written query sent to/in client class instance


*Parameters --*

data_path: None (`str`)

String that is the file path name where images are stored (default is set to none/assumption is to use current working directory)

new_folders: True (`bool`)

Option to create new folder/subdirectory to store images in current working directory (default is set to true)


*Returns --*

None 

Executes dynamic convolutional neural network model training/testing and outputs resulting summary/metrics/associated plots

#### encode_text ####

``` python
client.encode_text(self, dataset, text)
```

Encodes text found in dataset into tokenized sequences for usage by text-related ML models


*Parameters --* 

dataset

Data to be analyzed that is initialized in the client class instance (is set by default to dataframe format)

text: `str`

Text that comes from the document/dataset which is being analyzed


*Returns --* 

result: `[int, int, ..., int]` 

List of sequence of integers representing the individual identified text parts of the whole text

#### predict_text_sentiment ####

``` python
client.predict_text_sentiment(self, dataset, text)
```

Wrapper function used for repeated caching of standard backend processing/prediction tasks needed for sentiment analysis when sentiment analysis text classification query is called


*Parameters --* 

dataset

Data to be analyzed that is initialized in the client class instance (is set by default to dataframe format)

text: `str`

Text that comes from the document/dataset which is being analyzed


*Returns --* 

`classes.get(tf.keras.backend.get_value(prediction)[0])`

Retrieves classes of sequences of text predicted by the sentiment analysis LSTM model

#### text_classification_query ####

``` python
client.text_classification_query(self,
                                 instruction,
                                 preprocess=True,
                                 test_size=0.2,
                                 random_state=49,
                                 epochs=10,
                                 maxTextLength=200,
                                 generate_plots=True)
```

Generates LSTM model used to solve sentiment analysis text classification problem request from written query sent to/in client class instance


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

preprocess: True (`bool`)

Option to preprocess the data (is set by default to true)

test_size: 0.2 (`float`)

Percentage measure of the proportion of the data to include in test split of the dataset (is set by default to 0.2)

random_state: 49 (`int`)

Random value assigned that ensures the shuffling of the dataset before applying the test/training split remains the same across multiple runs (is set by default to 49)

epochs: 10 (`int`)

Number of full cycles used for model fitting/training on the training dataset (is set by default to 10)

maxTextLength: 200 (`int`)

Maximum length in terms of characters per sequence found in text of dataset initialized in client class instance (is set by default to 200)

generate_plots: True (`bool`)

Option to generate all relevant plots for query (is set by default to true)


*Returns --* 

None

Executes document summarization training/testing and outputs/stores resulting summary/metrics/associated plots

#### get_summary #### 

``` python
client.get_summary(self, text)
```

Wrapper function used for repeated caching of standard backend processing/prediction tasks needed for text summarization when text summarization query is called 


*Parameters --* 

text: `str`

Text that comes from the document/dataset which is being analyzed


*Returns --*

predictions: `[str, str,..., str]`

List of decoded predicted summarizations of the document being summarized

#### summarization_query ####

``` python
client.summarization_query(self, 
                           instruction,
                           preprocess=True,
                           test_size=0.2,
                           random_state=49,
                           epochs=1,
                           generate_plots=True)
```

Generates/stores contextual summmary of textual data in document using T5 pretrained text-to-text transformers from written query sent to/in the client class instance  


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

preprocess: True (`bool`)

Option to preprocess the data (is set by default to true)

test_size: 0.2 (`float`)

Percentage measure of the proportion of the data to include in test split of the dataset (is set by default to 0.2)

random_state: 49 (`int`)

Random value assigned that ensures the shuffling of the dataset before applying the test/training split remains the same across multiple runs (is set by default to 49)

epochs: 1 (`int`)

Number of full cycles used for model fitting/training on the training dataset (is set by default to 1)

generate_plots: True (`bool`)

Option to generate all relevant plots for query (is set by default to true)


*Returns --* 

None

Executes document summarization model training/testing and outputs resulting summary/metrics/associated plots

#### dimensionality_reducer ####

``` python
client.dimensionality_reducer(self, instruction)
```

Chooses and applies dimensionality reduction algorithm to the dataset provided based on instruction from written query sent to/in the client class instance


*Parameters --* 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance


*Returns --*

None

Applies dimensionality_reduc function to dataset instead of creating output

#### show_plots ####

``` python
client.show_plots(self, model)
```

Displays all plots generated by the model chosen based from the queries written in the client class instance


*Parameters --* 

model: `object`

Model selected based of the query written in the class client instance


*Returns --* 

None 

Simply prints all the plots present in the client class instance as the output

#### model_data ####

``` python
client.show_plots(self, model)
```

Displays all model keys visible in the model dictionary established in the client class instance


*Parameters --* 

model: `object`

Model selected based of the query written in the class client instance


*Returns --* 

None 

Simply prints out the list of model keys present in the model dictionary during client class instance

#### operators ####

``` python
client.operators(self, model)
```

Displays all operators applicable to the model dictionary established in the client class instance


*Parameters --* 

model: `object`

Model selected based of the query written in the class client instance


*Returns --* 

None 

Simply prints out the list of all operators defined and built-in for the models in the model dictionary in the client class instance 

#### accuracy ####

``` python
client.accuracy(self, model)
```

Displays all accuracy scores for the model chosen based from the queries written in the client class instance


*Parameters --* 

model: `object`

Model selected based of the query written in the class client instance


*Returns --* 

`self.models[model]['accuracy']`: `float`

Retrieves float value representing accuracy performance metric score for model

`{cross_val_score: self.models[model]['cross_val_score']}`: `float`

Retrieves float value representing cross_validation performance metric score for model


#### losses ####

``` python
client.losses(self, model)
```

Displays all loss validation metrics for the model chosen based from the queries written in the client class instance


*Parameters --* 

model: `object`

Model selected based of the query written in the class client instance


*Returns --* 

`self.models[model]['losses']`: float

Retrieves float value representing loss validation function performance metrics for model


