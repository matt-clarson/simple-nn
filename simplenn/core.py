import numpy as np
import functools

import simplenn.softmax as softmax
import simplenn.helpers as helpers

def train(X, y, num_classes, **params):
    hidden_size = params.setdefault('hidden_size', 100)
    iterations = params.setdefault('iterations', 10000)
    step_size = params.setdefault('step_size', 1.0)
    reg_strength = params.setdefault('reg_strength', 1e-3)

    W1, b1 = helpers.init_weights(X.shape[1], hidden_size)
    W2, b2 = helpers.init_weights(hidden_size, num_classes)

    for i in range(iterations):
        hidden_layer, output_layer = evaluate(X, W1, b1, W2, b2, cache=True)
    
        loss, probs = softmax.softmax_loss(
            output_layer, y, reg_strength,
            [W1, W2]
        )

        if i % 1000 == 0 :
            print(f'loss at iteration {i}: {loss}')
    
        d_W1, d_b1, d_W2, d_b2 = helpers.backpropagate(
            output_layer, hidden_layer, X, y, probs,
            W1, b1, W2, b2,
            reg_strength
        )

        W1, b1, W2, b2 = helpers.update_weights(
            [W1, b1, W2, b2], [d_W1, d_b1, d_W2, d_b2],
            step_size
        )
    
    return (W1, b1), (W2, b2)

def evaluate(X, W1, b1, W2, b2, cache=False):
    hidden_layer = helpers.activate(X.dot(W1) + b1)
    output_layer = hidden_layer.dot(W2) + b2
    return (hidden_layer, output_layer) if cache else output_layer

