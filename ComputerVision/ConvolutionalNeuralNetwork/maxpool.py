import numpy as np


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
