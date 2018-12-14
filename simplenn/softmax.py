import numpy as np
import functools

def softmax_loss(scores, target, reg_strength, weights):
    num_examples = scores.shape[0]
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probabilities[range(num_examples), target])
    data_loss = np.sum(correct_logprobs) / num_examples
    accumulate_reg_loss = lambda acc, W: acc + (0.5*reg_strength*np.sum(W*W))
    loss = functools.reduce(accumulate_reg_loss, weights, data_loss)
    return loss, probabilities

def softmax_diff(scores, target, probabilities):
    probabilities[range(scores.shape[0]), target] -= 1
    return probabilities / scores.shape[0]
