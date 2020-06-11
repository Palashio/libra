<img src="data/libra-logo.png" width="100%">

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
* [tuner.py](#tuner)
   * [class CNNHyperModel](#class-CNNHyperModel)
      * [__init__](#__init__)
      * [build](#build)
   * [tuneReg](#tuneReg)
   * [build_model](#build_model)
   * [tuneClass](#tuneClass)
   * [tuneCNN](#tuneCNN)  
* [generatePlots.py](#generatePlots)
  * [generate_clustering_plots](#generate_clustering_plots)
  * [generate_regression_plots](#generate_regression_plots)
  * [generate_classification_plots](#generate_classification_plots)
  * [generate_classification_together](#generate_classification_plots)
  * [plot_loss](#plot_loss)
  * [plot_corr](#plot_corr)
  * [plot_acc](#plot_acc)
* [NLP_preprocessing.py](#NLP_preprocessing)
  * [lemmatize_text](#lemmatize_text)
  * [tokenize_text](#tokenize_text)
  * [text_clean_up](#text_clean_up)
* [data_preprocesser.py](#data_preprocesser)
* [data_reader.py](#data_reader)
* [dimensionality_red_queries.py](#dimensionality_red_queries)
* [predictionQueries.py](#predictionQueries)
