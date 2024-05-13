import torch
from torch import device, cuda, tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = device('cuda' if cuda.is_available() else 'cpu')


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


class VaeAutoencoder(nn.Module):
    def __init__(self, dim_encoding):
        super(VaeAutoencoder, self).__init__()

        self.dim_encoding = dim_encoding
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        self.linear_mean = nn.Linear(2048, dim_encoding)
        self.linear_logvar = nn.Linear(2048, dim_encoding)

        self.linear = nn.Linear(dim_encoding, 2048)
        self.reshape = Reshape(-1, 128, 4, 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1)
        self.trim = Trim()

    def reparameterized(self, mean, var):
        eps = torch.randn(mean.size(0), mean.size(1)).to(self.device)
        z = mean + eps * torch.exp(var / 2.)
        return z

    def encode(self, x):  # Using silu instead of relu here
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = self.conv3(x)
        x = self.flatten(x)
        mean = self.linear_mean(x)
        var = self.linear_logvar(x)
        z = self.reparameterized(mean, var)
        return mean, var, z

    def decode(self, z):
        z = self.linear(z)
        z = self.reshape(z)
        z = F.silu(self.deconv1(z))
        z = F.silu(self.deconv2(z))
        z = F.silu(self.deconv3(z))
        z = self.trim(z)
        z = F.sigmoid(z)
        return z

    def forward(self, x):
        mean, var, z = self.encode(x)
        z = self.decode(z)
        return mean, var, z

    def train_model(
            self,
            training_data,
            batch_size=64,
            beta=0.000075,
            epochs=5,
            learning_rate=0.001
    ) -> tuple[nn.Module, list, list, list]:
        global train_loss_KL, train_loss_reconstruction, train_loss_total
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        train_losses_total, train_losses_total_avg = [], []
        train_losses_reconstruction, train_losses_reconstruction_avg = [], []
        train_losses_KL, train_losses_KL_avg = [], []

        for epoch in range(epochs):
            train_loss_total = 0.0
            train_loss_reconstruction = 0.0
            train_loss_KL = 0.0

            for input, _ in training_dataloader:
                input = input.to(device)
                mean, var, outputs = self(input)

                optimizer.zero_grad()
                loss1 = criterion(outputs, input)
                loss2 = beta * torch.mean(
                    -0.5 * torch.sum(1 + var - mean ** 2 - torch.exp(var), axis=1),
                    axis=0
                )
                # print("Time")
                # print(loss1)
                # print(loss2)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    train_losses_reconstruction.append(loss1.item())
                    train_losses_KL.append(loss2.item())
                    train_losses_total.append(loss.item())
            print("Finished epoch: ", epoch + 1)

        return (
            self.to('cpu'),
            train_losses_total,
            train_losses_KL,
            train_losses_reconstruction
        )

    def generate_data(self, n_samples=32) -> tensor:
        """
        Generates random data samples (of size n) from the latent space
        """
        device = next(self.parameters()).device
        input_sample = torch.randn(n_samples, self.dim_encoding).to(device)
        assert input_sample.shape[0] == n_samples
        return self.decode(input_sample)
