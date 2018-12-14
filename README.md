# simple-nn
A simple, handwritten Neural Network

## Install
Install with pip:

```bash
pip install git+git://github.com/matt-clarson/simple-nn.git#egg=simple-nn
```

## Use
To use, just import the `simplenn` package.

```python
import simplenn as snn

weights = snn.train(X, y, num_classes, **params)

snn.evaluate(X, *weights)
```

## Docs
The `simplenn` package provides two methods:

 * `train`
 * `evaluate`

### Train
The `train` method runs a simple, two layer neural network on a given dataset.

#### Params
Param | Required (y/N) | Type | Default | Use
---|---|---|---|---
`X` | Yes | `numpy.ndarray` | N/A | The training data to use
`y` | Yes | `numpy.ndarray` | N/A | The expected labels to train for
`num_classes` | Yes | `int` | N/A | The number of unique labels in the dataset
`hidden_size` | No | `int` | `100` | The size of the hidden layer
`iterations` | No | `int` | `10000` | The number of iterations to train for
`step_size` | No | `float` | `1.0` | The step size hyperparameter
`reg_strength` | No | `float` | `1e-3` | The strength of regularisation

#### Returns
The trained weights to use for evaluation as an array.

```python
X.shape # (2, 100)
y.shape # (100,)

W1, b1, W2, b2 = snn.train(X, y, 3, hidden_size=150)

W1.shape # (100, 150)
b1.shape # (1, 150)
W2.shape # (150, 3)
b2.shape # (1, 3)
```

### Classify
The `classify` method takes a set of trained weights and uses them to classify inputs of the same kind as the training data


