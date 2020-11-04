import gzip

import numpy as np


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
