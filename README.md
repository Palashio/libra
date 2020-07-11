<div align="center">
  

![Test Image 1](/tools/data/gh_images/new-logo.png)

## Fully Automated Machine Learning in One-Liners

[![Build Status](https://travis-ci.org/Palashio/libra.svg?branch=master)](https://travis-ci.org/Palashio/libra)
[![Downloads](https://pepy.tech/badge/libra)](https://pepy.tech/project/libra)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/the-libra-team/shared_invite/zt-ek6bpd47-hdIxXlRAenKfy5JNWe8bgw)
[![PyPi](https://img.shields.io/badge/pypi%20package-0.0.9-blue)](https://pypi.org/project/libra/)
[![Release](https://img.shields.io/badge/Next%20Release-July%2012-green)](https://pypi.org/project/libra/)


Libra is a deep learning API that allows users to use machine learning in their workflows in fluent one-liners. It is written in Python and TensorFlow and makes training neural networks as simple as a one line function call. It was written to make deep learning as simple as possible to every user. 
*** 

</div>

## Installation

Install latest release version:

```
pip install -U libra
```

Install directory from github:

```
git clone https://github.com/Palashio/libra.git
cd libra
pip install .
```
From Conda:

```python
conda install libra -c conda-forge
```

## Usage: the basics

The core functionality of libra works through the `client` object. A new client object should be created for every dataset that you want to produce results for. All information about the models that're built, the plots that are generated, and the metrics are created will be stored in the object.

You can then call different queries on that client object, and the dataset you passed to it will be used. 

```python
from libra import client

newClient = client('path/to/dataset') 
newClient.neural_network_query('please model the median number of households')
```
Now, calling 
```python
newClient.info()
```
will return a dictionary of all the information that was generated: 

```python
dict_keys(['id', 'model', 'num_classes', 'plots', 'target', 'preprocesser', 
          'interpreter', 'test_data', 'losses', 'accuracy'])
```

Other queries can also be called on the same object, and will be appended to the `models` dictionary.

```python
newClient.svm_query('predict the proximity to the ocean')
newClient.models().keys()

dict_keys(['regression_ANN', svm'])
```
