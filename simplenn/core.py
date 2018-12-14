import numpy as np
import functools

def init_weights(m, n):
    return 0.01 * np.random.randn(m, n), np.zeros((1, n))

def activate(n):
    """ ReLU activation function """
    return np.maximum(0, n)

def softmax_loss(scores, target, reg_strength, weights):
    num_examples = scores.shape[0]
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probabilities[range(num_examples), target])
    data_loss = np.sum(correct_logprobs) / num_examples
    accumulate_reg_loss = lambda acc, W: acc + (0.5*reg_strength*np.sum(W*W))
    return functools.reduce(accumulate_reg_loss, weights, data_loss)


def train(X, y, num_classes, **params):
    hidden_size = params.setdefault('hidden_size', 100)
    iterations = params.setdefault('iterations', 1e+5)
    step_size = params.setdefault('step_size', 1.0)
    reg_strength = params.setdefault('reg_strength', 1e-3)

    W1, b1 = init_weights(X.shape[1], hidden_size)
    W2, b2 = init_weights(hidden_size, num_classes)

    hidden_layer = activate(X.dot(W1) + b1)
    output_layer = hidden_layer.dot(W2) + b2

    loss = softmax_loss(output_layer, y, reg_strength, [W1, W2])
    print(f'loss: {loss}')

    return (W1, b1), (W2, b2)
