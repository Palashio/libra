<div align="center">
  

![Test Image 1](/tools/data/gh_images/libra-logo.png)

## Fully Automated Machine Learning in One-Liners

[![Build Status](https://travis-ci.org/Palashio/libra.svg?branch=master)](https://travis-ci.org/Palashio/libra)
[![Downloads](https://pepy.tech/badge/libra)](https://pepy.tech/project/libra)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/the-libra-team/shared_invite/zt-ek6bpd47-hdIxXlRAenKfy5JNWe8bgw)
[![PyPi](https://img.shields.io/badge/pypi%20package-0.0.1-blue)](https://pypi.org/project/libra/)
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

A new client object should be created for every dataset that you're working with. This is where the results of your models for the dataset will be stored.

You can then call different queries on that client object, and the dataset you passed to it will be used. 

```python
from libra.queries import client

newClient = client('path/to/dataset') 
newClient.neural_network_query('please model the median number of households')
```
