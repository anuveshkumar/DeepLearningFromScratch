import gzip

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# NETWORK FUNCTIONS

def convolution(image, filters, bias, stride=1):
    """Convolves filter over image using stride """

    # drop last value since it's a square matrix
    (num_filter, num_filter_channel, filter_dim, _) = filters.shape  # filter dimension
    num_image_channel, image_dim, _ = image.shape  # image dimensions

    output_dim = int((image_dim - filter_dim) / stride) + 1  # calculate the output dimension

    # ensure that the filter channel matches the image channels
    assert num_filter_channel == num_image_channel, "Ensure that the number of filter channels match the input channels"

    out = np.zeros((num_filter, output_dim, output_dim))

    # convolve each filter over the image
    for current_filter in range(num_filter):
        current_y = output_y = 0
        # move filter vertically across the image
        while current_y + filter_dim <= image_dim:
            current_x = output_x = 0
            # move filter horizontally across the image
            while current_x + filter_dim <= image_dim:
                # perform the convolution operations and add the bias
                out[current_filter, output_x, output_y] = np.sum(
                    filters[current_filter] * image[:, current_y: current_y + filter_dim,
                                              current_x: current_x + filter_dim]) + bias[current_filter]
                current_x += stride
                output_x += 1
            current_y += stride
            output_y += 1

    return out


def maxpool(image, kernel_size=2, stride=2):
    """Downsample input image using a kernel size of 'f' and stride of 's'"""
    num_channels, height_previous, width_previous = image.shape

    height = int((height_previous - kernel_size) / stride) + 1
    width = int((width_previous - kernel_size) / stride) + 1

    # create a matrix to hold the output of max pooling
    downsampled = np.zeros((num_channels, height, width))

    for i in range(num_channels):
        current_y = output_y = 0
        while current_y + kernel_size < height_previous:
            current_x = output_x = 0
            while current_x + kernel_size < width_previous:
                downsampled[i, output_y, output_x] = np.max(
                    image[i, current_y: current_y + kernel_size, current_x: current_x + kernel_size])
                current_x += stride
                output_x += 1

            current_y += stride
            output_y += 1

    return downsampled


def softmax(raw_predictions):
    """ pass raw predictions through softmax function"""
    return np.exp(raw_predictions) / np.sum(raw_predictions)


def categoricalcrossentropy(probabilities, labels):
    return - np.sum(labels * np.log(probabilities))


def initialize_filter(size, scale=1.0):
    """Initialize filter using a normal distribution with a std deviation
    inversely proportional to the square root of the number of units """
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initialize_weight(size):
    """ Initialize scale with a random normal distribution """
    return np.random.standard_normal(size) * 0.01


# DATA FUNCTIONS


def training_data():
    """Extract images by reading the file bytestream. Resaahpe the read values into a 3D matrix of
    dimensions [m, height, width] where m is the number of training examples"""
    with gzip.open(filename="ConvolutionalNeuralNetworks/t10k-images-idx3-ubyte.gz") as bytestream:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(bytestream.read(4), 'big')
        print(magic_number)
        # second 4 bytes is the number of images
        image_count = int.from_bytes(bytestream.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(bytestream.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(bytestream.read(4), 'big')

        image_data = bytestream.read()
        data = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_count, row_count, column_count)
        print(data.shape)


def training_labels():
    with gzip.open(filename="ConvolutionalNeuralNetworks/t10k-labels-idx1-ubyte.gz") as bytestream:
        # first 4 bytes is the magic number
        magic_number = int.from_bytes(bytestream.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(bytestream.read(4), 'big')
        label_data = bytestream.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


if __name__ == "__main__":
    training_data()
    labels = training_labels()
    print(labels)

    # image = Image.open("ConvolutionalNeuralNetworks/some_bird.jpg")
    # image_numpy = np.array(image)
    # image_numpy = image_numpy.transpose((2, 0, 1))
    #
    # filters = np.random.random((image_numpy.shape[0], 5, 5))
    # filters = filters.reshape(1, filters.shape[0], filters.shape[1], filters.shape[2])
    #
    # convolved = convolution(image_numpy, filters=filters, bias=np.zeros(1), stride=1)
    # pooled = maxpool(convolved)
    #
    # (nf2, dim2, _) = pooled.shape
    # fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer
