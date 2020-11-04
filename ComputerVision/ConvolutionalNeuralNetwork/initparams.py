import numpy as np


def initialize_filter(size, scale=1.0):
    """Initialize filter using a normal distribution with a std deviation
    inversely proportional to the square root of the number of units """
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initialize_weight(size):
    """ Initialize scale with a random normal distribution """
    return np.random.standard_normal(size) * 0.01