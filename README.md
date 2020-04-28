# Libra: deep learning in fluent one-liners

![Image description](libra.png)

A high-level machine learning API written in Python and Tensorflow that makes training deep neural networks as simple as a one-line function call. 

Table of Contents
=================

* [Prediction Queries: building blocks](#queries)
   * [Regression Neural Network](#regression-neural-network-query)
   * [Classification Neural Network](#classification-neural-network-query)
   * [K-Means Clustering](#k-means-clustering)
   * [Nearest Neighbors](#nearest-neighbors)
   * [Support Vector Machines](#support-vector-machine)
   * [Decision Tree](#decision-tree)
* [Image Generation](#image-generation)
   * [Class Wise Image Generation](#class-wise-image-generation)
   * [Convolutional Neural Networks](#convolutional-neural-network)
* [Model Information](#model-information)
   * [Model Tuning](#model-tuning)
   * [Plotting](#plotting)
   * [Dataset Information](dataset-information)
* [Dimensionality Reduction](#dimensionality-reduction)
   * [Reduction Pipeliner](#reduction-pipeliner)
   * [Principle Component Analysis](#principle-component-analysis)
   * [Feature Importances via Random Forest Regressor](feature-importances-via-random-forest-regressor)
   * [Independent Component Analysis](#indepedent-component-analysis)
* [Process Logger](#process-logger)
* [Providing Instructions](#instructions)

## Queries ##

### Regression Neural Network Query ###

Let's start with the most basic query. This will build a feed-forward network for a class that you specify
```python
import libra

newClient = client('dataset')
newClient.SingleRegressionQuery('Model the median house value')
```
No preprocessing is neccesary. All plots, losses, and models are stored in the models field in the client class. 

Basic tuning with the number of layers is done when you call this query. If you'd like to automatically more in depth you can call: 

```python
newClient.tune('regression', inplace = False)
```

Calling this tunes hyperparameters like node count, layer count, learning rate, and other features. This will return the best network and if ```inplace = True``` it will replace it in the client class under ```regression_ANN```. 

### Classification Neural Network Query ###

```python
newClient = client('dataset')
newClient.classificationQuery('Predict building name')
newClient.tune('classification')
```
This creates a neural network to predict building names given your dataset. Any number of classes will work for this query. Note that ```categorical_crossentropy``` and an `adam` optimizer is used as a default. This can be changed as well. 

### K-means Clustering ###

```python
newClient = client('dataset')
newClient.kMeansClusteringQuery()
```

This will create a k-means clustering algorithm trained on your processed dataset. It continues to grow the number of clusters until the ``inertia`` value stops decreasing by atleast 1000 units. If you'd like to specify the number of clusters you'd like it to use you can do ``clusters = number_of_clusters``. 


### Nearest-neighbors ###

```python
newClient = client('dataset')
newClient.nearestNeighborQuery()
```

This will use scikit's learns nearest neighbor function to return the best nearest neighbor model on the dataset. Values are stored under the ```nearest_neighbor``` field in the model dictionary. 

### Support Vector Machine ###

```python
newClient = client('dataset')
newClient.svmQuery('Model the value of houses')
```

This will use scikit's learns SVM function to return the best support vector machine on the dataset. Values are stored under the ```svm``` field in the model dictionary. 

NOTE: A linear kernel is used as the default, this can be modified by specifying:

```newClient.svmQuery('rbf_kernel')```

### Decision Tree ###

```python
newClient = client('dataset')
newClient.svmQuery()
```

This will use scikit's learns Decision Tree function to return the best decision tree on the dataset. Values are stored under the ```decision_tree``` field in the model dictionary. 

```newClient.decisionTreeQuery('Model the type of cars')```

If you'd like to modify hyperparameters feed them this is the order in which you should provide the parameters:

```decisionTreeQuery(instruction, max_depth, min_samples_split, max_samples_split, min_samples_leaf, max_samples_leaf)```
***

## Image Generation ##

### Class wise image generation ### 
If you want to generate an image dataset to use in one of your models you can do:

```python
generateSet('apples', 'oranges', 'bananas', 'pineapples')
```

This will create seperate folders in your directory with each of these names with around ~100 images for each class. An updated version of Google Chrome is required for this feature. If you'd like to use it with an older version of Chrome please install the appropriate chromedriver. 

### Convolutional Neural Network ###
If you'd like to generate images and fit it automatically to a CNN you can use this command:

```python
newClient.classGenQuery('apples', 'oranges')
```
This will generate a dataset of apples and oranges by parsing google images, prepprocess the dataset appropriately and then fit it to a Convolutional Neural Network. All images are reduced to a standard (224, 224, 3) size using a traditional OpenCV resizing algorithm. Default size is 100 images for each. You can specify this size by adding ```class_size = number_in_each_class```

Note: all images will be resized to (224, 224, 3). Properties are maintained by using a geometric image transformation explained here: 

[OpenCV Transformation](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html)

If you'd like to perform tuning on the CNN you can do this: 

```python
newClient.tuneCNN()
```
Unlike other functions, tuning on the CNN is not done automatically because of how computationally intensive it is. 

***

## Model Information ## 

### Model Tuning ###

In order to further tune your model (this might take a while), you can call: 

```python
newClient.tune('regression')
```
This will tune:
  1. Number of Layers
  2. Number of Nodes in every layer
  3. Learning Rate
  4. Activation Functions
  
In order to ensure that the tuned models accuracy is robust, every model is ran multiple times and the accuracy is averaged. This ensures that the model configuration is truly the best. 

### Plotting ###
All plots are stored during runtime. This function plots all generated graphs for your current client object on one pane. 

```python
newClient.plotAll('regression')
```

If you'd like to extract a single plot, you can do:

```python
newClient.show_plots('regression')
``` 

and then 

```python
newClient.getModels()['regression']['plots']['trainlossvstestloss']
```

### Dataset Information ###
In depth metrics about your dataset and similarity information can be generated by calling:

```python
newClient.stat_analysis()
```
A information graph as well as a similarity spectrum shown below will be generated:

![Image description](similarity.png)

This represents 5 columns that have the smallest cosine distance: these might need to be removed to reduce noise. You can specify whether you want to remove with ```inplace = True```. 

If you'd like information on just one column you can do: 

```python
 newClient.stat_analysis(dataset[columnname])
```

***

## Dimensionality Reduction ##

### Reduction Pipeliner ###

If you'd like to get the best pipeline for dimensionality reduction you call can call:

```python
 dimensionalityReduc("I want to estimate number of crime", path_to_dataset) 
```

Libra current supports feature importance identifier using random forest regressor, indepedent component analysis, and principle component analysis. The output of this function should look something like this: 

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
The baseline accuracy represents the accuracy acheived without any dimensionality reduction techniques. Then, each possible permutation of reduction technique is displayed with its respective accuracy. At the bottom is the best pipeline which resulted in the highest accuracy.

If you'd like to replace the dataset with one that replaces it with the best reduced one, you can just specify ```inplace=True```.

### Principle Component Analysis ###

Performing Principle Component is as simple as: 

```python 
dimensionalityPCA("Estimating median house value", path_to_dataset)
```

NOTE: this will select the optimal number of principal components to keep. If you'd like to specify the number of components you can just do ```n_components = number_of_components```. 

### Feature Importances via Random Forest Regressor ###
Using the random forest regressor to identify feature importances is as easy as calling: 

```python
dimensionalityRF("Estimating median house value", path_to_dataset)
```
This will find the optimal number of features to use and will return the dataset with the best accuracy. If you'd like to manually set the number of feature you can do ```n_features = number of features```. 

### Indepedent Component Analysis ###

Performing Indepedent Component Analysis is as simple as: 

```python 
dimensionalityICA("Estimating median house value", path_to_dataset)
```

If this does not converge a message will be displayed for users to warn them. 

***

## Process Logger ##

Libra will automatically output the current process running in a hierarchial format like this:

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

A quiet mode feature will be implemented after package conversion is completed.

### Instructions ###

Libra uses intelligent part of speech recognition to analyze user instructions and match it with a column in user datasets. 
  1. [Textblob](https://textblob.readthedocs.io/en/dev/), a part of speech recognition algorithm, is used to identify parts of speech.
  2. Self-developed part of speech deciphering algorithm is used to extract relevant parts of a sentence.
  3. Levenshentein distances are used to match relevant parts of the sentence to a column name.
  4. That column name is run through an LSTM Recurrent Neural Network to obtain latent representation.
  5. Both instruction and column latent representation is compared to verify. 
