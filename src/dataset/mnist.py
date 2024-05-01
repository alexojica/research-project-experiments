import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array, to_categorical


def prepare_mnist_dataset(train_ratio=0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (default_x_train, default_y_train), (default_x_test, default_y_test) = tf.keras.datasets.mnist.load_data()

    # Concatenate the training and testing sets
    x = np.concatenate([default_x_train, default_x_test])
    y = np.concatenate([default_y_train, default_y_test])

    # Split the combined dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_ratio))

    # reduce images to vector of pixels. 28×28 image will be 784 pixel input values
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    # # reduce memory requirements by forcing precision of the pixel values to be 32 bit
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    print('Training images size: ', x_train.shape[0])
    print('Training input size:', x_train.shape[1])
    print('Training labels size:', y_train.shape[0])

    print('Testing images size: ', x_test.shape[0])
    print('Testing input size:', x_test.shape[1])
    print('Testing labels size:', y_test.shape[0])

    # normalize input (pixels are gray scale between 0 and 255)
    x_train = x_train / 255
    x_test = x_test / 255
    print('Normalized result', x_train.shape, x_test.shape)

    # One-hot encoding of output: transform the vector of class integers into a label binary variables
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('Output shape', y_train.shape)
    return x_train, x_test, y_train, y_test
