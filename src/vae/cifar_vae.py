from src.utils import kl_loss, vae_loss_fn, vae_classifier_loss_fn

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self):
        super(Trim, self).__init__()

    def forward(self, x):
        return x[:, :, :32, :32]


class VaeAutoencoderClassifier(nn.Module):
    def __init__(self, dim_encoding):
        super(VaeAutoencoderClassifier, self).__init__()

        self.dim_encoding = dim_encoding
        self.latent_space_vector = None
        self.encodings = None
        self.z_dist = None
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        self.linear_mean = nn.Linear(2048, 100)
        self.linear_logvar = nn.Linear(2048, 100)

        self.linear = nn.Linear(100, 2048)
        self.reshape = Reshape(-1, 128, 4, 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1)
        self.trim = Trim()

    def encode(self, x):
        """
        Uses Silu instead of ReLu?
        """
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = self.conv3(x)
        x = self.flatten(x)
        return self.reparameterize(x)

    def decode(self, z):
        z = self.linear(z)
        z = self.reshape(z)
        z = F.silu(self.deconv1(z))
        z = F.silu(self.deconv2(z))
        z = F.silu(self.deconv3(z))
        z = self.trim(z)
        z = F.sigmoid(z)
        return z

    def reparameterize(self, encodings: Tensor) -> Tensor:
        mu = encodings[:, :self.dim_encoding]

        # must do exponential, otherwise get value error that not all positive
        sigma = torch.exp(encodings[:, self.dim_encoding:])
        z_dist = Normal(mu, sigma)
        self.z_dist = z_dist
        z = z_dist.rsample()
        return z

    def forward(self, x):
        # sample latent code z from q given x.
        encodings = self.encoder(x)
        self.encodings = encodings
        z = self.reparameterize(encodings)

        assert z.shape[1] == self.dim_encoding
        self.latent_space_vector = z
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return z

    # def train_model(
    #         self,
    #         training_data,
    #         batch_size=64,
    #         alpha=1.0,
    #         beta=1.0,
    #         epochs=5,
    #         learning_rate=0.01
    # ) -> tuple[nn.Module, list]:
    #     complete_loss_fn = vae_classifier_loss_fn(alpha, beta)
    #
    #     vae_classifier_model = self.to('cuda')
    #     optimizer = torch.optim.Adam(
    #         params=vae_classifier_model.parameters(),
    #         lr=learning_rate
    #     )
    #
    #     training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    #
    #     total_losses = []
    #
    #     for epoch in range(epochs):
    #         for input, labels in training_dataloader:
    #             input = input.to('cuda')
    #             labels = labels.to('cuda')
    #             output = vae_classifier_model(input)
    #
    #             # loss function to back-propagate on
    #             loss = complete_loss_fn(
    #                 input,
    #                 output,
    #                 self.z_dist,
    #                 labels
    #             )
    #
    #             # back propagation
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #         print("Finished epoch: ", epoch + 1)
    #     return (
    #         vae_classifier_model.to('cpu'),
    #         total_losses
    #     )