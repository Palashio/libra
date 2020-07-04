## INITIAL BUILD RELEASE DATE: JULY 16TH.
in progress....this README.md is out of date. Documentation site coming soon. 

![image](tools/data/gh_images/verve-logo.png)

# verve: Deep Learning fluent in one-liners

verve is a deep learning API that allows users to use machine learning in their workflows in fluent one-liners. It is written in Python and TensorFlow and makes training neural networks as simple as a one line function call. It was written to make deep learning as simple as possible to every user. 
*** 

## Guiding Principles ## 
  * **Beginner Friendly.** verve is an API designed to be used by developers with no deep learning experience whatsoever. It is built so that users with no knowledge in preprocessing, modeling, or tuning can build high-performance models with ease without worrying about the details of implementation.
  
  * **Quick Integration.** With the recent rise of machine learning on the cloud, the developer community has failed to make easy-to-use platforms that exist locally and integrate directly into workflows. verve allows users to develop models directly in programs with hundreds of API endpoints without having to worry about the transition to the cloud.
  
  * **Automation.** End-to-end pipelines containing hundreds of processes are automatically run for the user. The developer only has to consider what they want to accomplish from the task and the location of their initial dataset.
  
  * **Easy Extensibility.** Queries are split into standalone modules. Under the dev-pipeline module you can pipeline both different and new modules and integrate them into the workflow directly. This allows newly developed features to be easily tested before integrating them into the main program. 

***

Table of Contents
=================

* [Prediction Queries: building blocks](#queries)
   * [Regression Neural Network](#regression-neural-network)
   * [Classification Neural Network](#classification-neural-network)
   * [Convolutional Neural Network](#convolutional-neural-network)
   * [K-Means Clustering](#k-means-clustering)
   * [Nearest Neighbors](#nearest-neighbors)
   * [Support Vector Machines](#support-vector-machine)
   * [Decision Tree](#decision-tree)
* [Image Generation](#image-generation)
   * [Class Wise Image Generation](#class-wise-image-generation)
   * [Generate Dataset & Convolutional Neural Network](#generate-dataset-and-convolutional-neural-network)
* [Model Information](#model-modifications)
   * [Model Tuning](#model-tuning)
   * [Plotting](#plotting)
   * [Dataset Information](#dataset-information)
* [Dimensionality Reduction](#dimensionality-reduction)
   * [Reduction Pipeliner](#reduction-pipeliner)
   * [Principle Component Analysis](#principle-component-analysis)
   * [Feature Importances via Random Forest Regressor](feature-importances-via-random-forest-regressor)
   * [Independent Component Analysis](#indepedent-component-analysis)
* [Process Logger](#process-logger)
* [Pipelining for Contributors](#pipelining-for-contributors)
* [Providing Instructions](#instructions)
* [Example Projects](#example-projects)

## Queries ##

Generally, all queries have the same structure. You should always be passing an English instruction to the query. The information that you generate from the query will always be stored in the `client`class in the model's dictionary. When you call a query on the `client` object, an instruction will be passed. Any format will be decoded, but avoiding more complex sentence structures will yield better results. If you already know the exact target class label name, you can also provide it. 

### Regression Neural Network ###

Let's start with the most basic query. This will build a feed-forward network for a continuous label that you specify.

```python
import verve

newClient = client('dataset')
newClient.regression_query_ann('Model the median house value')
```
No preprocessing is neccesary. All plots, losses, and models are stored in the models field in the client class.

Basic tuning with the number of layers is done when you call this query. If you'd like to tune more in depth you can call: 

```python
newClient.tune('regression', inplace = False)
```
To specify which model to tune, you must pass the type of model that you'd like to perform tuning on. 

This function tunes hyperparameters like node count, layer count, learning rate, and other features. This will return the best network and if ```inplace = True``` it will replace the old model it in the client class under ```regression_ANN```. 

### Classification Neural Network ###

This query will build a feed-forward neural network for a classification task. As such, your label must be a discrete variable. 

```python
newClient = client('dataset')
newClient.classification_query_ann('Predict building name')
```
This creates a neural network to predict building names given your dataset. Any number of classes will work for this query. By default, ```categorical_crossentropy``` and an `adam` optimizer are used. 

### Convolutional Neural Network ###
Creating a convolutional neural network for a dataset you already have created is as simple as: 

```python
newClient = client()
newClient.convolutional_query('path_to_class1', 'path_to_class2', 'path_to_class3')
```
For this query, no initial shallow tuning is performed is done because of how memory intensive CNN's can be. User specified parameters for this query are currently being implemented. The defaults can be found in the `prediction_queries.py` file.

### K-means Clustering ###

This query will create a k-means clustering algorithm trained on your processed dataset. 

```python
newClient = client('dataset')
newClient.kmeans_clustering_query()
```

It continues to grow the number of clusters until the value of ``inertia`` stops decreasing by at least 1000 units. This is a threshold determined based on several papers, and extensive testing. This can also be changed by specifying ```threshold = new_threshold_num```. If you'd like to specify the number of clusters you'd like it to use you can do ``clusters = number_of_clusters``. 


### Nearest-neighbors ###

This query will use scikit-learn's nearest-neighbor function to return the best nearest neighbor model on the dataset.

```python
newClient = client('dataset')
newClient.nearest_neighbor_query()
```

You can specify the ```min_neighbors, max_neighbors``` as keyword arguments to the function. Values are stored under the ```nearest_neighbor``` field in the model dictionary. 

### Support Vector Machine ###

This will use scikit-learn's SVM function to return the best support vector machine on the dataset.

```python
newClient = client('dataset')
newClient.svm_query('Model the value of houses')
```

Values are stored under the ```svm``` field in the model dictionary. 

NOTE: A linear kernel is used as the default, this can be modified by specifying your new kernel name as a keyword argument: ```kernel = 'rbf_kernel'```. 


### Decision Tree ###

This will use scikit's learns decision tree function to return the best decision tree on the dataset.

```python
newClient = client('dataset')
newClient.decision_tree_query()
```

This will use scikit's learns Decision Tree function to return the best decision tree on the dataset. Values are stored under the ```decision_tree``` field in the model dictionary. 

You can specify these hyperparameters by passing them as keyword arguments to the query: ```max_depth = num, min_samples_split = num, max_samples_split = num, min_samples_leaf = num, max_samples_leaf= num)```
***

## Image Generation ##

### Class wise image generation ### 
If you want to generate an image dataset to use in one of your models you can do:

```python
generate_set('apples', 'oranges', 'bananas', 'pineapples')
```

This will create separate folders in your directory with each of these names with ~100 images for each class. An updated version of Google Chrome is required for this feature; if you'd like to use it with an older version of Chrome please install the appropriate chromedriver. 

### Generate Dataset and Convolutional Neural Network ###
If you'd like to generate images and fit it automatically to a Convolutional Neural Network you can use this command:

```python
newClient.generate_fit_cnn('apples', 'oranges')
```
This particular will generate a dataset of apples and oranges by parsing Google Images, preprocess the dataset appropriately and then fit it to a convolutional neural network. All images are reduced to a standard (224, 224, 3) size using a traditional OpenCV resizing algorithm. Default size is the number of images in one Google Images page *before* having to hit more images, which is generally around 80-100 images. 

The infrastructure to generate more images is currently being worked on. 

Note: all images will be resized to (224, 224, 3). Properties are maintained by using a geometric image transformation explained here: [OpenCV Transformation](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html).

***

## Model Modifications ## 

### Model Tuning ###

In order to further tune your neural network models, you can call: 

```python
newClient.tune('convolutional neural network')
```
This will tune:
  1. Number of Layers
  2. Number of Nodes in every layer
  3. Learning Rate
  4. Activation Functions
    
In order to ensure that the tuned models accuracy is robust, every model is run multiple times and the accuracies are averaged. This ensures that the model configuration is optimal.

You can just specify what type of network you want to tune — it will identify your target model from the `models` dictionary using another instruction algorithm. 

NOTE: Tuning for CNN's is **very** memory intensive, and should not be done frequently. 

### Plotting ###
All plots are stored during runtime. This function plots all generated graphs for your current client object on one pane. 

```python
newClient.plot_all('regression')
```

If you'd like to extract a single plot, you can do:

```python
newClient.show_plots('regression')
``` 

and then 

```python
newClient.getModels()['regression']['plots']['trainlossvstestloss']
```
No other plot retrieval technique is currently implemented. While indexing nested dictionaries might seem tedious, this was allowed for fluency. 

### Dataset Information ###
In depth metrics about your dataset and similarity information can be generated by calling:

```python
newClient.stat_analysis()
```
A information graph as well as a similarity spectrum shown below will be generated:

![Image description](data/similarity.png)

This represents 5 columns that have the smallest cosine distance; you might need to remove these columns because they're too similar to each other and will just act as noise. You can specify whether you want to remove them with ```inplace = True```. Information on cosine similarity can be found [here](https://www.sciencedirect.com/topics/computer-science/cosine-similarity).

If you'd like information on just one column you can do: 

```python
 newClient.stat_analysis(dataset[columnname])
```
***

## Dimensionality Reduction ##

### Reduction Pipeliner ###

If you'd like to get the best pipeline for dimensionality reduction you can call:

```python
 dimensionality_reduc("I want to estimate number of crime", path_to_dataset) 
 
```
Instructions like "I want to model x" are provided in the dimensionality reduction pipeline because it identifies which prediction objective you would like to maximize the accuracy for. Providing this instruction helps verve provide users with the best modification pipeline. 

verve current supports feature importance identification using random forest regressor, indepedent component analysis, and principle component analysis. The output of the dimensionalityReduc() function should look something like this: 

```
Baseline Accuracy: 0.9752906976744186
----------------------------
Permutation --> ('RF',) | Final Accuracy --> 0.9791666666666666
Permutation --> ('PCA',) | Final Accuracy --> 0.8015988372093024
Permutation --> ('ICA',) | Final Accuracy --> 0.8827519379844961
Permutation --> ('RF', 'PCA') | Final Accuracy --> 0.3316375968992248
Permutation --> ('RF', 'ICA') | Final Accuracy --> 0.31419573643410853
Permutation --> ('PCA', 'RF') | Final Accuracy --> 0.7996608527131783
Permutation --> ('PCA', 'ICA') | Final Accuracy --> 0.8832364341085271
Permutation --> ('ICA', 'RF') | Final Accuracy --> 0.8873546511627907
Permutation --> ('ICA', 'PCA') | Final Accuracy --> 0.7737403100775194
Permutation --> ('RF', 'PCA', 'ICA') | Final Accuracy --> 0.32630813953488375
Permutation --> ('RF', 'ICA', 'PCA') | Final Accuracy --> 0.30886627906976744
Permutation --> ('PCA', 'RF', 'ICA') | Final Accuracy --> 0.311531007751938
Permutation --> ('PCA', 'ICA', 'RF') | Final Accuracy --> 0.8924418604651163
Permutation --> ('ICA', 'RF', 'PCA') | Final Accuracy --> 0.34205426356589147
Permutation --> ('ICA', 'PCA', 'RF') | Final Accuracy --> 0.9970639534883721

Best Accuracies
----------------------------
["Permutation --> ('ICA', 'PCA', 'RF) | Final Accuracy --> 0.9970639534883721"]

```
The baseline accuracy represents the accuracy acheived without any dimensionality reduction techniques. Then, each possible reduction technique permutation is displayed with its respective accuracy. At the bottom is the pipeline which resulted in the highest accuracy. You can also specify which of the reduction techniques you'd like to try by passing ```reducers= ['ICA', 'RF']``` to the function.

If you'd like to replace the dataset with one that replaces it with the best reduced one, you can just specify ```inplace=True```.

### Principle Component Analysis ###

Performing Principle Component is as simple as: 

```python 
dimensionality_PCA("Estimating median house value", path_to_dataset)
```

NOTE: this will select the optimal number of principal components to keep. The default search space is up to the number of columns in your dataset. If you'd like to specify the number of components you can just do ```n_components = number_of_components```.  

### Feature Importances via Random Forest Regressor ###
Using the random forest regressor to identify feature importances is as easy as calling: 

```python
dimensionality_RF("Estimating median house value", path_to_dataset)
```
This will find the optimal number of features to use and will return the dataset with the best accuracy. If you'd like to manually set the number of feature you can do ```n_features = number of features```. 

### Indepedent Component Analysis ###

Performing Indepedent Component Analysis can be done by calling:

```python 
dimensionality_ICA("Estimating median house value", path_to_dataset)
```

If this does not converge a message will be displayed for users to warn them by default.  

***

## Process Logger ##

verve will automatically output the current process running in a hierarchial format like this:

```
loading dataset...
  |
  |- getting most similar column from instruction...
    |
    |- generating dimensionality permutations...
      |
      |- running each possible permutation...
        |
        |- realigning tensors...
          |
          |- getting best accuracies...
 ```

A quiet mode feature is currently being implemented. 

***

## Pipelining for Contributors ##

In order to help make verve extensible, a process pipeliner has been implemented to help contributors easily test their newly-developed modules. 

Let's say you've developed a different preprocesser for data that you want to test before integrating it into verve's primary workflow. This is the process to test it out:

First, you want to initialize your base parameters, which are your instructions, the path to your dataset, and any other information your new function might require.

```
init_params = {
    'instruction': "Predict median house value",
    'path_to_set': './data/housing.csv',
}
```

You can then modify the main pipeline: 

<pre>
single_regression_pipeline = [initializer,
                <b>your_own_preprocessor</b>, #is originally just preprocessor
                instruction_identifier,
                set_splitter,
                modeler,
                plotter]
</pre>

These pipelines can be found under the ``dev-pipeliner`` folder. Currently, this format is only supported for the single regression pipeline. Complete integration of pipelining into the main framework is currently being implemented. 

Finally, you can run your pipeline by using: 

```
[func(init_params) for func in reg_pipeline] 

```

All model information should be stored in ```init_params```. If you'd like to modify smaller details, you can copy over the module and modify the smaller detail; this split was not done to maintain ease of use of the pipeline. 

***
## Instructions ##

verve uses intelligent natural language processing to analyze user instructions and match it with a column in user datasets. 
  1. [Textblob](https://textblob.readthedocs.io/en/dev/), a part of speech recognition algorithm, is used to identify parts of speech.
  2. A self-developed part-of-speech deciphering algorithm is used to extract relevant parts of a sentence. 
  3. Masks are generated to represent all words as tensors in order for easy comparison
  4. Levenshentein distances are used to match relevant parts of the sentence to a column name.
  5. Target column selected based on lowest levenshentein distance and is returned.

## Example Projects ##

To get started, take a look at some of these examples of data science projects analyzing datasets using verve. 
