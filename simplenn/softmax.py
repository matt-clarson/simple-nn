import numpy as np
import functools

stabilise_softmax_input = lambda f: f - np.max(f)

def softmax_loss(scores, target):
    num_examples = scores.shape[0]
    scores = stabilise_softmax_input(scores)
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probabilities[range(num_examples), target])
    loss = np.sum(correct_logprobs) / num_examples
    return loss, probabilities

def softmax_diff(scores, target, probabilities):
    probabilities[range(scores.shape[0]), target] -= 1
    return probabilities / scores.shape[0]
