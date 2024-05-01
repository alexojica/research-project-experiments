import os
import sys

module_to_import = os.path.dirname(sys.path[0])
print(module_to_import)
sys.path.append(module_to_import)

from utils import sample

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

MNIST_INPUT_SIZE = 784

HIDDEN_LAYER_SIZE = 512

DEFAULT_DIMENSION_ENCODING = 2


class VaeEncoder(nn.Module):
    def __init__(self, dim_encoding):
        """
        dim_encoding - dimensionality of the latent space
        encoder outputs two parameters per dimension of the latent space, which is typical for VAEs
        """
        super(VaeEncoder, self).__init__()

        # linear layer that takes in MNIST input size
        self.fc1 = nn.Linear(MNIST_INPUT_SIZE, HIDDEN_LAYER_SIZE)

        # linear layer that takes output of fc1 and transforms it to dim_encoding
        # TODO: why dim_encoding * 2
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, dim_encoding * 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        takes in input of tensor x and outputs tensor of the latent space vectors.

        For example, given that x has 6 data points and VAE has latent space of 2 dimensions:

        x: torch.Size([6, 1, 28, 28])
        output: torch.Size([6, 2])
        """
        # reshapes each data point tensor from its original shape to a 1D tensor from 2nd dimension onwards
        # e.g. torch.Size([6, 1, 28, 28]) -> torch.Size([6, 784])
        x = torch.flatten(x, start_dim=1)

        # pass through fc1 followed by ReLU activation
        x = F.relu(self.fc1(x))

        # absence of an activation function means that the output can be any real-valued number
        return self.fc2(x)


class VaeDecoder(nn.Module):
    """
    Decoder that outputs 28x28 pixel images from the latent space vectors
    """
    def __init__(self, dim_encoding):
        super(VaeDecoder, self).__init__()

        # linear layer that takes latent space vectors as input
        self.fc1 = nn.Linear(dim_encoding, HIDDEN_LAYER_SIZE)

        # linear layer that outputs to MNIST input size
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, MNIST_INPUT_SIZE)

    def forward(self, x: Tensor) -> Tensor:
        """
        Takes in Tensor of the latent space vectors and outputs a 28x28 pixel image

        For example, given 6 data points as input and 2-dimensional latent space:
        - x: torch.Size([6, 2])
        - output: torch.Size([6, 1, 28, 28])
        """
        # pass through fc1 followed by ReLU activation, resulting in x.shape: torch.Size([6, 512])
        x = F.relu(self.fc1(x))

        # sigmoid activation function to map the output to a range between 0 and 1, resulting
        # in x.shape: torch.Size([6, 784])
        x = torch.sigmoid(self.fc2(x))

        # match input shape back to 28x28 pixels
        return x.reshape(-1, 1, 28, 28)


class VaeClassifierDecoder(nn.Module):
    """
    Classifier decoder that outputs both images and its corresponding vector of label probabilities
    """
    def __init__(self, dim_encoding):
        super(VaeClassifierDecoder, self).__init__()
        self.fc1 = nn.Linear(dim_encoding, HIDDEN_LAYER_SIZE)

        # outputs both an image plus a 10-element vector for digit classification
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, MNIST_INPUT_SIZE + 10)

    def forward(self, x: Tensor) -> Tensor:
        """
        Takes in Tensor of the latent space vectors and outputs the 28x28 pixel image and vector of label probabilities

        For example, given 6 data points as input and 2-dimensional latent space:
        - x: torch.Size([6, 2])
        - output: torch.Size([6, 1, 28, 28]), torch.Size([6, 10])
        """
        # pass through fc1 followed by ReLU activation, resulting in x.shape: torch.Size([6, 512])
        x = F.relu(self.fc1(x))

        # sigmoid activation function to map the output to a range between 0 and 1, resulting
        # in x.shape: torch.Size([6, 794])
        x = torch.sigmoid(self.fc2(x))
        return x


class VaeAutoencoder(nn.Module):
    """
    Variational Autoencoder. VAEs extend the concept of AEs by mapping the input data to a distribution
    (usually a multivariate normal distribution). Generates data by sampling from the learned latent space.

    Returns a tensor of a random MNIST image.
    """
    def __init__(self, dim_encoding):
        super(VaeAutoencoder, self).__init__()
        self.dim_encoding = dim_encoding
        self.encoder = VaeEncoder(dim_encoding)
        self.decoder = VaeDecoder(dim_encoding)

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns a tensor of a random MNIST image.
        """
        encodings = self.encoder(x)
        self.encodings = encodings

        # maps the encodings to a distribution and generate samples from it
        sampled = sample(encodings)
        return self.decoder(sampled)


class VaeAutoencoderClassifier(nn.Module):
    """
    Classifier decoder that returns both images and its corresponding vector of label probabilities
    """
    def __init__(self, dim_encoding):
        super(VaeAutoencoderClassifier, self).__init__()
        self.dim_encoding = dim_encoding
        self.encoder = VaeEncoder(dim_encoding)
        self.decoder = VaeClassifierDecoder(dim_encoding)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns two tensors: a MNIST image and its probabilities of labels
        """
        # Tensor of 2-dimension encodings in the latent space
        # e.g. if given 6 data points, encoded is torch.Size([6, 2])
        encodings = self.encoder(x)

        self.encodings = encodings

        # maps the encoded vector to a distribution and generate samples from it
        sampled = sample(encodings)
        decoded = self.decoder(sampled)
        return decoded[:, :MNIST_INPUT_SIZE].reshape(-1, 1, 28, 28), decoded[:, MNIST_INPUT_SIZE:]
