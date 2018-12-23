import numpy as np
import functools

import simplenn.softmax as softmax

def initialise_weights(network_shape):
    return [np.random.randn(network_shape[i-1], n) * 0.001
                for i, n in enumerate(network_shape)
                if i != 0]

def take_mini_batch_sample(inputs, expected_outputs, batch_size):
    mini_batch_indexes = set()
    while len(mini_batch_indexes) < batch_size:
        mini_batch_indexes.add(np.random.randint(inputs.shape[0]))
    selection = list(mini_batch_indexes)
    return inputs[selection], expected_outputs[selection]

def populate_hidden_layers(X, weights):
    layers = []
    for i, W in enumerate(weights[:-1]):
        input_layer = X if i == 0 else layers[-1][0]
        hidden = hidden_layer(input_layer, W)
        layers.append(hidden)
    return layers

def hidden_layer(input_layer, weights):
    output = np.maximum(0, input_layer.dot(weights))
    def backprop(prev_diff):
        prev_diff[output <= 0] = 0
        d_input = prev_diff.dot(weights.T)
        d_weights = input_layer.T.dot(prev_diff)
        return d_input, d_weights
    return output, backprop

def output_layer(input_layer, weights, expected_output):
    output = input_layer.dot(weights)
    data_loss, probs = softmax.softmax_loss(output, expected_output)
    def backprop():
        diff = softmax.softmax_diff(output, expected_output, probs)
        d_input = diff.dot(weights.T)
        d_weights = input_layer.T.dot(diff)
        return d_input, d_weights
    return data_loss, output, backprop

def regularise(weights, reg_strength):
    L2_reg = lambda acc, x: acc + (0.5 * reg_strength * np.sum(x**2))
    return functools.reduce(L2_reg, weights, 0)

def backpropagate(layers):
    prev_diff = None
    d_weights = []
    for i, layer in enumerate(reversed(layers)):
        backprop = layer[1]
        d_input, d_W = backprop() if i == 0 else backprop(prev_diff)
        prev_diff = d_input
        d_weights.append(d_W)

    return reversed(d_weights)

def normalise_gradients(gradients):
    return [dW / np.max(np.abs(dW)) for dW in gradients]

