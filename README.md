# Libra
A machine learning API that makes training deep neural networks as simple as a one-line function call. 

# Usage: the basics
Fitting a model to a feed-forward neural network to any dataset is as simple as this:
```python
newClient = client('dataset')
newClient.SingleRegressionQuery('Model the median house value')
newClient.tune('regression')
```
No preprocessing is neccesary. All plots, losses, and models are stored in the models field in the client class. Calling tune tunes hyperparameters like number of layers, learning rate, and layer size. 
