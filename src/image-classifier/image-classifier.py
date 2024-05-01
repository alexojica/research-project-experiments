import sys
import os

import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

module_to_import = os.path.join(os.path.dirname(sys.path[0]), 'dataset')
print(module_to_import)
sys.path.append(module_to_import)

from mnist import prepare_mnist_dataset


def neural_network(num_classes, num_pixels) -> tf.keras.Sequential:
    model = Sequential()

    # input layer
    model.add(Input(shape=(num_pixels,)))

    # hidden layer
    model.add(Dense(units=256, activation='relu'))

    # hidden layer
    model.add(Dense(units=64, activation='relu'))

    # output layer
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    return model


def train_model(
        model: tf.keras.Sequential,
        x_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate=0.001,
        epochs=5,
        batch_size=32
):
    # Optimizer: SGD (Stochastic Gradient Descent)
    opt = SGD(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


def test_model(
        model: tf.keras.Sequential,
        x_test: np.ndarray,
        y_test: np.ndarray
):
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Error: ", (100 - scores[1] * 100), "%")


x_train, x_test, y_train, y_test = prepare_mnist_dataset()
nn_model = neural_network(10, 784)
train_model(nn_model, x_train, y_train)
test_model(nn_model, x_test, y_test)
