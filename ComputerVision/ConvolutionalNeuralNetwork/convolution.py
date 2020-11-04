import numpy as np


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
