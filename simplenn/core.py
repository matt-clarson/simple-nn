import numpy as np
import functools

import simplenn.helpers as helpers

def train(X, y, num_classes, **params):
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    iterations = params.setdefault('iterations', 10000)
    step_size = params.setdefault('step_size', 0.01)
    reg_strength = params.setdefault('reg_strength', 1e-3)
    network_shape = [
        X.shape[1],
        *params.setdefault('network_shape', [100]),
        num_classes
    ]
    batch_size = params.setdefault('batch_size', 256)
    weights = helpers.initialise_weights(network_shape)
    for i in range(iterations):
        batch_input, batch_output = helpers.take_mini_batch_sample(
            X, y, batch_size
        )
        layers = helpers.populate_hidden_layers(batch_input, weights)
    
        softmax_loss, output, output_backprop = helpers.output_layer(
            layers[-1][0], weights[-1], batch_output
        )
        loss = softmax_loss + helpers.regularise(weights, reg_strength)

        if i % 1000 == 0:
            print(f' loss at iteration {i}: {loss}')
        layers.append((output, output_backprop))
    
        d_weights = helpers.backpropagate(layers)
        d_weights = helpers.normalise_gradients(d_weights)
        weights = [weights[l] - step_size*d_W
                    for l, d_W in enumerate(d_weights)]
    return weights

def classify(X, *weights):
    prev_layer = np.hstack((X, np.ones((X.shape[0], 1))))
    for W in weights[:-1]:
        prev_layer = np.maximum(0, prev_layer.dot(W))
    output = prev_layer.dot(weights[-1])
    return np.argmax(output, axis=1)

