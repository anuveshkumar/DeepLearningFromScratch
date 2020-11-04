import numpy as np


def softmax(raw_predictions):
    """ pass raw predictions through softmax function"""
    return np.exp(raw_predictions) / np.sum(raw_predictions)
