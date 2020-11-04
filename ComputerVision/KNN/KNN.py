import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def l1_distance(X1, X2):  # also known as manhattan distance
    # sum(mod(X1 - X2))
    # assert X1.shape == X2.shape, "matrices must have the same dimensions"
    return np.sum(np.abs(X1 - X2))  # np.sum calculate total distance in each channel and sum them up


def l2_distance(X1, X2, axis=None):  # also known as euclidean distance
    # assert X1.shape == X2.shape, "matrices must have the same dimensions"
    return np.sqrt(np.sum(np.square(X1 - X2), axis=axis))


def calculate_distance_vector(trained_images, test_image):
    """
    calculate the distance vector between the train set and the test image
    :param trained_images:  dimension, N x H x W x C
    :param test_image: H X W X C
    :return: N X 1
    """
    return l2_distance(trained_images, test_image, axis=(1, 2, 3))


def find_nearest_neighbour_class(distance_vector, class_vector, k=1):
    ranked_vector = np.argsort(distance_vector)
    top_k_classes = np.array([x for _, x in sorted(zip(distance_vector, class_vector))])
    classes, counts = np.unique(top_k_classes, return_counts=True)
    max_index = np.argmax(counts)

    return classes[max_index]


def unpickle(file):
    with open(file, 'rb') as foo:
        dict = pickle.load(foo, encoding='latin1')
    return dict


def view_image(image_vector, label):
    image = np.reshape(image_vector, newshape=(32, 32, 3), order='F').transpose((1, 0, 2))
    plt.imshow(image)
    plt.title(label)
    plt.show()


def train(X, Y):
    train_X = X
    train_Y = Y
    return train_X, train_X


def evaluate(test_image):


if __name__ == "__main__":
    file = "../Datasets/cifar-10-python/cifar-10-batches-py/data_batch_1"
    dataset = unpickle(file)
    print(dataset.keys())
    # print(image_data[0].reshape(32, 32, 3).shape)
    # print(dataset['batch_label'])
    # print(dataset['filenames'])

    image_data = dataset['data']
    labels = dataset['labels']
    view_image(image_data[3], labels[3])

    # split the data into train, test and validation set
    X_train, X_test, Y_train, Y_test = train_test_split(image_data, labels, test_size=0.2, random_state=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)




# NUM_SAMPLES = 100
# trained_images = np.random.random((NUM_SAMPLES, 32, 32, 3))
# trained_classes = np.random.randint(low=1, high=10, size=NUM_SAMPLES)
# test_image = np.random.random((32, 32, 3))
#
# x = calculate_distance_vector(trained_images, test_image)
# label = find_nearest_neighbour_class(x, k=5)
# print(label)
