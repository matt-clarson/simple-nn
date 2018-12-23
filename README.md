# simple-nn
A simple, handwritten Neural Network

#### Contents
 * [Install](#install)
 * [Use](#use)
 * [Docs](#docs)
   * [preprocess](#preprocess)
   * [train](#train)
     * [Params](#params)
     * [Returns](#returns)
     * [Network Shape](#network-shape)
   * [classify](#classify)


## Install
Install with pip:

```bash
pip install git+git://github.com/matt-clarson/simple-nn.git
```

## Use
To use, just import the `simplenn` package.

```python
import simplenn as snn

weights = snn.train(X, y, num_classes, **params)

snn.evaluate(X, *weights)
```

## Docs
The `simplenn` package provides three methods:

 * `preprocess`
 * `train`
 * `classify`

### Preprocess
The `preprocess` method performs some simple preprocessing on datasets. It zero-centres the data, and then normalises it to the range: `-1 <= x <= 1`.

While not strictly necessary, this method should be used on inputs before they are given to the `train` or `classify` methods, to minimise the numerical stability of the operations those methods employ.

### Train
The `train` method runs a simple, softmax neural network on a given dataset.

#### Params
Param | Required (y/N) | Type | Default | Use
---|---|---|---|---
`X` | Yes | `numpy.ndarray` | N/A | The training data to use
`y` | Yes | `numpy.ndarray` | N/A | The expected labels to train for
`num_classes` | Yes | `int` | N/A | The number of unique labels in the dataset
`iterations` | No | `int` | `10000` | The number of iterations to train for
`step_size` | No | `float` | `1.0` | The step size hyperparameter
`reg_strength` | No | `float` | `1e-3` | The strength of regularisation
`network_shape` | No | `list` | `[100]` | A list describing the shape of the neural network - see below
`batch_size` | No | `int` | `256` | The size of mini batches to take of the input data when performing gradient descent

#### Returns
The trained weights to use for evaluation as an array.

```python
X.shape # (2, 100)
y.shape # (100,)

W1, b1, W2, b2 = snn.train(X, y, 3, hidden_size=[150])

W1.shape # (100, 150)
W2.shape # (150, 3)
```

#### Network Shape
The network shape expresses the depth and "width" of the neural network. The length of the given list is the number of hidden layers to use in the network. THe list should be a list of integers, where each integer is the size of that hidden layer. So for example:

```python
snn.train(X, y, num_classes, network_shape=[100, 50])
```

Would train a network with two hidden layers and one output layer (i.e. a 3 layer network), where the first hidden layer has the shape `(X.shape[1], 100)`, and the second layer has the shape `(100, 50)`.

### Classify
The `classify` method takes a set of trained weights and uses them to classify inputs of the same kind as the training data

