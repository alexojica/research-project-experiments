import sys
import os

dataset_path = os.path.join(os.path.dirname(sys.path[0]), 'src', 'dataset')
sys.path.append(dataset_path)
image_classifier_path = os.path.join(os.path.dirname(sys.path[0]), 'src', 'image-classifier')
sys.path.append(image_classifier_path)

from mnist import prepare_mnist_dataset
from image_classifier import neural_network, train_model, test_model


x_train, x_test, y_train, y_test = prepare_mnist_dataset()
nn_model = neural_network(10, 784)
train_model(nn_model, x_train, y_train)
test_model(nn_model, x_test, y_test)
