import numpy as np

import simplenn.softmax as softmax

def init_weights(m, n):
    return 0.01 * np.random.randn(m, n), np.zeros((1, n))

def update_weights(weights, gradients, step_size):
    return [weights[i] - step_size * gradients[i] for i in range(len(weights))]

def activate(n):
    """ ReLU activation function """
    return np.maximum(0, n)

def evaluate(X, *weights, cache=False):
    W1, b1, W2, b2 = weights
    hidden_layer = activate(X.dot(W1) + b1)
    output_layer = hidden_layer.dot(W2) + b2
    return (hidden_layer, output_layer) if cache else output_layer

def backpropagate(output_layer, hidden_layer, X, y, probs, W1, b1, W2, b2, reg):
    d_output = softmax.softmax_diff(output_layer, y, probs)
    
    d_W2 = hidden_layer.T.dot(d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)

    d_hidden = d_output.dot(W2.T)
    d_hidden[hidden_layer <= 0] = 0

    d_W1 = X.T.dot(d_hidden)
    d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

    d_W2 += reg * W2
    d_W1 += reg * W1

    return d_W1, d_b1, d_W2, d_b2
