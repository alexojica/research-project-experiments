from src.utils import kl_loss, vae_loss_fn, vae_classifier_loss_fn

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal


class VaeAutoencoderClassifier(nn.Module):
    def __init__(self, channel_num, kernel_num, dim_encoding):
        super().__init__()
        self.latent_space_vector = None
        self.z_dist = None
        self.encodings = None
        self.dim_encoding = dim_encoding
        self.channel_num = channel_num
        self.kernel_num = kernel_num

        # encoder
        self.encoder = nn.Sequential(
            self.conv_bloc(channel_num, kernel_num // 4),
            self.conv_bloc(kernel_num // 4, kernel_num // 2),
            self.conv_bloc(kernel_num // 2, kernel_num),
        )

        # decoder
        self.decoder = nn.Sequential(
            self.deconv_block(kernel_num, kernel_num // 2),
            self.deconv_block(kernel_num // 2, kernel_num // 4),
            nn.Linear(kernel_num // 4, kernel_num + 10),
            nn.Sigmoid()
        )

    def reparameterize(self, encodings: Tensor) -> Tensor:
        mu = encodings[:, :self.dim_encoding]

        # must do exponential, otherwise get value error that not all positive
        sigma = torch.exp(encodings[:, self.dim_encoding:])
        z_dist = Normal(mu, sigma)
        self.z_dist = z_dist
        z = z_dist.rsample()
        return z


    def conv_bloc(self, input_size, output_size):
        return nn.Sequential(
            nn.Conv2d(
                input_size, output_size,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
        )

    def deconv_block(self, input_size, output_size):
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_size, output_size,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        # sample latent code z from q given x.
        encodings = self.encoder(x)
        self.encodings = encodings
        z = self.reparameterize(encodings)

        assert z.shape[1] == self.dim_encoding
        self.latent_space_vector = z
        return self.decoder(z)

    def train_model(
            self,
            training_data,
            batch_size=64,
            alpha=1.0,
            beta=1.0,
            epochs=5,
            learning_rate=0.01
    ) -> tuple[nn.Module, list]:
        complete_loss_fn = vae_classifier_loss_fn(alpha, beta)

        vae_classifier_model = self.to('cuda')
        optimizer = torch.optim.Adam(
            params=vae_classifier_model.parameters(),
            lr=learning_rate
        )

        training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        total_losses = []

        for epoch in range(epochs):
            for input, labels in training_dataloader:
                input = input.to('cuda')
                labels = labels.to('cuda')
                output = vae_classifier_model(input)

                # loss function to back-propagate on
                loss = complete_loss_fn(
                    input,
                    output,
                    self.z_dist,
                    labels
                )

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Finished epoch: ", epoch + 1)
        return (
            vae_classifier_model.to('cpu'),
            total_losses
        )