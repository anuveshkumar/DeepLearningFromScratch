import numpy as np


def categoricalCrossEntropy(probabilities, labels):
    return - np.sum(labels * np.log(probabilities))