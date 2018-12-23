#! /usr/bin/env python
import json, pickle
import numpy as np
import simplenn

def load_training_data():
    with open('data/training_images.pkl', 'rb') as f:
        training_images = pickle.loads(f.read())
    with open('data/training_labels.pkl', 'rb') as f:
        training_labels = pickle.loads(f.read())
    return training_images, training_labels

def split_data(images, labels, split_size):
    validation_mask = []
    training_mask = list(range(images.shape[0]))
    while len(validation_mask) < split_size:
        n = training_mask.pop(np.random.randint(len(training_mask)))
        validation_mask.append(n)
    validation_set = images[validation_mask], labels[validation_mask]
    training_set = images[training_mask], labels[training_mask]
    return training_set, validation_set

def load_params():
    with open('simplenn-params.json', 'r') as f:
        params = json.loads(f.read())
    pretty_params = json.dumps(params, sort_keys=True, indent=4)
    return params, pretty_params

def main():
    print('Loading training data')
    data = load_training_data()
    print('Separating training and validation data')
    training, validation = split_data(*data, 10000)

    del data # free up memory

    params, pretty_params = load_params()
    print(f'Running with parameters: {pretty_params}')

    X, y = training
    X = simplenn.preprocess(X)

    weights = simplenn.train(X, y, 10, **params)

    X_v, y_v = validation
    X_v = simplenn.preprocess(X_v)
    predictions = simplenn.classify(X_v, *weights)
    accuracy = np.mean(predictions == y_v)

    print(f'accuracy: {accuracy}')

if __name__ == '__main__':
    main()
