<div align="center">

<img src="/tools/data/gh_images/logo.png" alt="drawing" width="100"/>
       
# Libra

**An ergonomic machine learning library for non-technical users. Save time. Blaze through ML.**

[![Build Status](https://travis-ci.org/Palashio/libra.svg?branch=master)](https://travis-ci.org/Palashio/libra)
[![Downloads](https://pepy.tech/badge/libra)](https://pepy.tech/project/libra)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/the-libra-team/shared_invite/zt-ek6bpd47-hdIxXlRAenKfy5JNWe8bgw)

[![PyPi](https://img.shields.io/badge/pypi%20package-1.0.0-blue)](https://pypi.org/project/libra/)
[![Release](https://img.shields.io/badge/Next%20Release-Sep%2012-green)](https://pypi.org/project/libra/)
[![Website shields.io](https://img.shields.io/website-up-down-blue-red/http/shields.io.svg)](https://libradocs.github.io//)
[![Issues](https://img.shields.io/github/issues/Palashio/libra)]()


</div>

---

## Trending Contributors

[![](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/images/0)](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/links/0)[![](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/images/1)](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/links/1)[![](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/images/2)](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/links/2)[![](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/images/3)](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/links/3)[![](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/images/4)](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/links/4)[![](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/images/5)](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/links/5)[![](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/images/6)](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/links/6)[![](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/images/7)](https://sourcerer.io/fame/anas-awadalla/Palashio/libra/links/7)


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

Alternatively you can build and use the docker image locally with:

```
docker build . -f docker/libra-normal/Dockerfile -t libra
docker run -v /path/to/my/data:/data -it --rm libra
```

Or if you have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed.

```
docker build . -f docker/libra-gpu/Dockerfile -t libra-gpu
docker run -v /path/to/my/data:/data --gpus all -it --rm libra-gpu
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
newClient.model().keys()

dict_keys(['regression_ANN', svm'])
```

## Tutorials

 - Full documentation can be found at [libradocs.github.io](https://libradocs.github.io/). 
 - A list of resources can be found on our [awesome-libra](https://github.com/Palashio/awesome-libra) repository. 

---
 

## Asking for help
Welcome to the Libra community!

If you have any questions, feel free to:
1. [read the docs](https://libradocs.github.io/).
2. [Search through the issues](https://github.com/Palashio/libra/issues?q=is%3Aissue+is%3Aclosed).
3. [Ask on stackoverflow](https://stackoverflow.com/questions/ask?guided=false) with the tag libra.
4. [Join our slack](https://join.slack.com/t/the-libra-team/shared_invite/zt-ek6bpd47-hdIxXlRAenKfy5JNWe8bgw).



## Demos

![alt-text](/tools/data/gh_images/gif.gif)

## Contact

Shoot me an email at [ps9cmk@virginia.edu](mailto:ps9cmk@virginia.edu) if you'd like to get in touch!

Follow me on [twitter](https://twitter.com/palashgshah) for updates and my insights about modern AI!
