# Libra Reference Manual  

Libra is an open source deep learning API that provides a toolkit of intuitive and easily accessible machine learning methods, enabling users to streamline their data science workflows. The reference manual contains a detailed description of the Libra API. This reference delineates each of the methods avaiable for use, explaining how one can utilize them as well their respective parameters. The manual assumes that you have a baseline comprehension of some of the key concepts in machine learning. The reference is organized mainly in terms of location in the code where the method was defined as well as in accordance with OOP hierarchy, with a class introduced first and then its methods appearing underneath.
***

Table of Contents
=================

* [dataset_labelmatcher.py](#dataset_labelmatcher)
   * [produceMask](#produceMask)
   * [get_similar_column](#get_similar_column)
   * [get_similar_model](#get_similar_model)
* [grammartree.py](#grammartree)
   * [get_value_instruction](#get_value_instruction)
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
  * [image_preprocess](#image_preprocess)
  * [processColorChanel](#processColorChanel)  
  * [process_dates](#process_dates)
  * [generate_column_labels](#generate_column_labels)
* [data_reader.py](#data_reader)
  * [class DataReader](#class-DataReader)
    * [init](#__init__-DataReader)
    * [retrieve_extension](#retrieve_extension)
    * [data_generator](#data_generator)
* [dimensionality_red_queries.py](#dimensionality_red_queries)
  * [dimensionality_reduc](#dimensionality_reduc)
  * [dimensionality_RF](#dimensionality_RF)
  * [dimensionality_PCA](#dimensionality_PCA)
  * [dimensionality_ICA](#dimensionality_ICA)
  * [get_last_file](#get_last_file)
* [predictionQueries.py](#predictionQueries)
  * [clearLog](#clearLog)
  * [logger](#logger)
  * [class client](#class-client)
    * [init](#__init__-client)
    * [get_models](#get_models)
    * [save](#save)
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

***

## dataset_labelmatcher ##

### produceMask ###

``` python
libra.produceMask(instruction_label)
```

Generates/stores with base object an encoding mask of form list ```[int/float, int/float, int/float]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- 

instruction_label: `str`

A string based of the instruction sent to encode in the form of a mask


Returns -- 

mask: `[int/float, int/float, int/float]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### get_similar_column ###

``` python
libra.get_similar_column(instruction, dataset)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### get_similar_model ###

``` python
libra.get_similar_model(model_requested, model_keys)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

***

## grammartree ##

### get_value_instruction ###

``` python
libra.get_value_instruction(sent)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

***

## regression_split_functions ##
   
### initializer ###

``` python
libra.initializer(params)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved
   
### preprocesser ###
   
``` python
libra.preprocesser(params)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### instruction_identifier ###

``` python
libra.instruction_identifier(params)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### set_splitter ###

``` python
libra.set_splitter(params)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved
   
### modeler ###

``` python
libra.modeler(params)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved
   
### plotter ###

``` python
libra.plotter(params)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

***

## predictionModelCreation ##

### get_keras_model_reg ###

``` python
libra.get_keras_model_reg(dataset, i)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### get_keras_model_class ###

``` python
libra.get_keras_model_class(dataset, i, num_classes)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved
   
### get_keras_text_class ###

``` python
libra.get_keras_text_class(max_features, num_classes)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved
   
### getKerasConvolutional ###

``` python
libra.instruction_getKerasConvolutional()
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### load_model ###

``` python
libra.load_model(self, load_path)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

***

## tuner ##

### class CNNHyperModel ###

``` python
class CNNHyperModel(HyperModel)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### __init__-CNNHyperModel ####

``` python
CNNHyperModel.__init__(self, input_shape, num_classes)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### build ####

``` python
CNNHyperModel.build(self, hp)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### tuneReg ###

``` python
libra.tuneReg(data, target)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### build_model ###

``` python
libra.build_model(hp)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### tuneClass ###

``` python
libra.tuneClass(X, y, num_classes)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### tuneCNN ###  

``` python
libra.tuneCNN(X, y, num_classes)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

***

## generatePlots ##

### generate_clustering_plots ###

``` python
libra.generate_clustering_plots(kmeans, dataPandas, dataset)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### generate_regression_plots ###

``` python
libra.generate_regression_plots(history, data, label)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### generate_classification_plots ###

``` python
libra.generate_classification_plots(history, data, label, model, X_test, y_test)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### generate_classification_together ###

``` python
libra.generate_regression_plots(history, data, model, X_test, y_test)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### plot_loss ###

``` python
libra.plot_loss(history)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### plot_corr ###


``` python
libra.plot_corr(data, data, col=[])
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### plot_acc ###

``` python
libra.plot_acc(history)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

***

## NLP_preprocessing ##

### get_target_values ###

``` python
libra.get_target_values(data, instruction, yLabel)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### lemmatize_text ###

``` python
libra.lemmatize_text(dataset)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved


### tokenize_text ###

``` python
libra.tokenize_text(dataset)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### text_clean_up ###

``` python
libra.text_clean_up(dataset)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### fix_slang ###

``` python
libra.fix_slang(text)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

***

## data_preprocesser ##

### initial_preprocesser ###

``` python
libra.initial_preprocesser(data, instruction, preprocess)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### structured_preprocesser ###

``` python
libra.structured_preprocesser(data)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### image_preprocess ###

``` python
libra.image_preprocess(data_path)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

``` python
libra.process_dates(data)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved### processColorChanel ###

``` python
libra.generate_regression_plots(history, data, label)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### process_dates ###

``` python
libra.process_dates(data)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### generate_column_labels ###

``` python
libra.generate_column_labels(pipeline, numeric_cols)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

***

## data_reader ##

### class-DataReader ###

``` python
class DataReader()
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved
    
#### __init__-DataReader ####

``` python
DataReader.__init__(self, filepath)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### retrieve_extension ####

``` python
DataReader.retrieve_extension(self)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved
    
#### data_generator ####

``` python
DataReader.data_generator(self)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

***

## dimensionality_red_queries ##

### logger ###

``` python
libra.logger(instruction, found="", slash="")
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation

Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved
 
### dimensionality_reduc ###

``` python
libra.dimensionality_reduc(instruction, dataset, arr = ["RF", "PCA", "ICA"], inplace = True)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved


### dimensionality_RF ###

``` python
libra.dimensionality_RF(instruction, dataset, target="", y="", n_features=10)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### dimensionality_PCA ###

``` python
libra.dimensionality_PCA(instruction, dataset, target="", y="", n_components=10)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved


### dimensionality_ICA ###

``` python
libra.dimensionality_ICA(instruction, dataset, target="", y="")
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved


### get_last_file ###

``` python
libra.get_last_file()
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

***

## predictionQueries ##

### clearLog ###

``` python
libra.clearLog()
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

### class-client ###

``` python
class client()
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### __init__-client ####

``` python
client.__init__(self, data)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### get_models ####

``` python
client.get_models(self, model_requested)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### save ####

``` python
client.save(self, model, save_model, save_path = os.getcwd())
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### predict ####

``` python
client.predict(self, modelKey, data)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

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

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

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

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

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
                                callback_mode='min',
                                maximizer="val_loss",
                                save_model=True,
                                save_path=os.getcwd())
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### kmeans_clustering_query ####


``` python
client.kmeans_clustering_query(self, 
                               instruction,
                               preprocess=True,
                               generate_plots=True,
                               drop=True,
                               save_path=os.getcwd())
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

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

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved
    
#### nearest_neighbor_query ####


``` python
client.nearest_neighbor_query(self, 
                              instruction,
                              preprocess=True,
                              drop=None,
                              min_neighbors=3,
                              max_neighbors=10)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### decision_tree_query ####

``` python
client.decision_tree_query(self, 
                           instruction,
                           preprocess=True,
                           drop=None,
                           test_size=0.2,
                           drop=True)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### allClassQuery ####

``` python
client.allClassQuery(self, 
                     instruction,
                     preprocess=True,
                     test_size=0.2
                     drop=None,
                     random_state=49,
                     save_model=1)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### tune ####

``` python
client.tune(self, model_to_tune)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### stat_analysis ####

``` python
client.stat_analysis(self, column_name="none", drop=None)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### convolutional_query ####

``` python
client.convolutional_query(self, data_path=None, new_folders=True)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### encode_text ####

``` python
client.encode_text(self, dataset, text)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### predict_text_sentiment ####

``` python
client.predict_text_sentiment(self, dataset, text)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

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

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- mask: `[int/long, int/long, int/long]` 

Outputs an list of encoded numerical representation of the instruction text recieved

#### get_summary #### 

``` python
client.get_summary(self, text)
```

Generates/stores with base object an encoding mask of form list ```[int/long, int/long, int/long]``` based on frequency of distinct characters used for sequence to sequence pre-training for natural language generation


Parameters -- instruction_label: `str`

Acquires a string based of the instruction sent to encode in the form of a mask


Returns -- 

mask: `[int/long, int/long, int/long]`



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

Generates/stores contextual summmary from textual data in document using T5 pretrained text-to-text transformers  

Parameters -- 

instruction: `str`

String that is the instruction from a written query sent to/in the client class instance

preprocess = True (`bool`)

Option to preprocess the data (is set by default to true)

test_size=0.2 (`float`)

Percentage measure of the proportion of the data to include in test split of the dataset (is set by default to 0.2)

random_state=49 (`int`)

Random value assigned that ensures the shuffling of the dataset before applying the test/training split remains the same across multiple runs (is set by default to 49)

epochs=1 (`int`)

Number of full cycles used for model fitting/training on the training dataset (is set by default to 1)

generate_plots=True (`bool`)

Option to generate all relevant plots for query (is set by default to 1)


Returns -- 

None

Executes document summarization training and outputs resulting summary/metrics/associated plots

#### dimensionality_reducer ####

``` python
client.dimensionality_reducer(self, instruction)
```

Chooses and applies dimensionality reduction algorithm to the dataset provided based on instruction given in query sent to the client class instance


Parameters -- 

instruction: `str`

A string that is the instruction from a written query sent to/in the client class instance


Returns --

None

Applies dimensionality_reduc function to dataset instead of creating output

#### show_plots ####

``` python
client.show_plots(self, model)
```

Displays all plots generated by the model chosen based from the queries written in the client class instance


# Parameters -- 

model: `object`

The model selected based of the query written in the class client instance


# Returns -- 

None 

Simply prints all the plots as the output of using the method
