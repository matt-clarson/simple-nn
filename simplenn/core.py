import numpy as np
import functools

import simplenn.softmax as softmax
import simplenn.helpers as helpers


def hidden_layer(input_layer, weights):
    output = np.maximum(0, input_layer.dot(weights))
    def backprop(prev_diff):
        prev_diff[output <= 0] = 0
        d_input = prev_diff.dot(weights.T)
        d_weights = input_layer.T.dot(prev_diff)
        return d_input, d_weights
    return output, backprop

def output_layer(input_layer, weights, expected_output, regularised_loss):
    output = input_layer.dot(weights)
    data_loss, probs = softmax.softmax_loss(output, expected_output)
    loss = data_loss + regularised_loss
    def backprop():
        diff = softmax.softmax_diff(output, expected_output, probs)
        d_input = diff.dot(weights.T)
        d_weights = input_layer.T.dot(diff)
        return d_input, d_weights
    return loss, output, backprop


def train(X, y, num_classes, **params):
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    iterations = params.setdefault('iterations', 10000)
    step_size = params.setdefault('step_size', 0.01)
    reg_strength = params.setdefault('reg_strength', 1e-3)
    network_shape = [X.shape[1]] + params.setdefault('network_shape', [100]) + [num_classes]

    initialise_weights = lambda m, n: np.random.randn(m, n) * 0.001
    weights = [initialise_weights(network_shape[i-1], n) for i, n in enumerate(network_shape) if i != 0]
    print(f'weights: {[W.shape for W in weights]}')
    for i in range(iterations):
        accumulator = lambda acc, x: acc + 0.5*reg_strength*np.sum(x*x)
        regularised_loss = functools.reduce(accumulator, weights, 0)
        layers = []
        for j, W in enumerate(weights[:-1]):
            layers.append(hidden_layer(X if j == 0 else layers[-1][0], W))
    
        loss, output, output_backprop = output_layer(layers[-1][0], weights[-1], y, regularised_loss)
        if i % 1000 == 0:
            print(f' loss at iteration {i}: {loss}')
        layers.append((output, output_backprop))
    
        d_weights = []
        prev_diff = None
        for k, layer in enumerate(reversed(layers)):
            backprop = layer[1]
            d_input, d_W = backprop() if k == 0 else backprop(prev_diff)
            prev_diff = d_input
            d_weights.append(d_W)
    
        weights = [weights[l] - step_size*d_W for l, d_W in enumerate(reversed(d_weights))]
    
    # end for
    return weights

def classify(X, *weights):
    prev_layer = np.hstack((X, np.ones((X.shape[0], 1))))
    for W in weights[:-1]:
        prev_layer = np.maximum(0, prev_layer.dot(W))
    output = prev_layer.dot(weights[-1])
    return np.argmax(output, axis=1)

