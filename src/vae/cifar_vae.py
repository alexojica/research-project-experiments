import torch
from torch import device, cuda, tensor, Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader

from src.utils import vae_loss_fn, kl_loss


device = device('cuda' if cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    """
    https://github.com/jaywonchung/Learning-ML/tree/master/Implementations/Conditional-Variational-Autoencoder
    """
    def __init__(self, dim_encoding):
        super().__init__()
        self.dim_encoding = dim_encoding

        # x: (N, 3, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        # x: (N, 64, 16, 21)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        # x: (N, 128, 8, 10)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        # x: (N, 256, 4, 4)
        self.fc4 = nn.Linear(in_features=256 * 4 * 4, out_features= 2 * dim_encoding)
        # x: (N, 2*latent_dim)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         m.bias.data.fill_(0.)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = x.view(x.shape[0], 256 * 4 * 4)
        return self.fc4(x)


class Decoder(nn.Module):
    def __init__(self, dim_encoding):
        super().__init__()
        self.dim_encoding = dim_encoding

        # z: (N, latent_dim)
        self.fc1 = nn.Linear(in_features=dim_encoding, out_features=448 * 2 * 2)
        self.bn1 = nn.BatchNorm1d(num_features=448 * 2 * 2)
        # z: (N, 448*2*2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=448, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        # z: (N, 256, 4, 4)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        # z: (N, 128, 8, 8)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        # z: (N, 64, 16, 16)

        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)
        # z: (N, 3, 32, 32)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         m.bias.data.fill_(0.)

    def forward(self, z):
        z = F.relu(self.bn1(self.fc1(z)))
        z = z.view(-1, 448, 2, 2)
        z = F.relu(self.bn2(self.deconv2(z)))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))
        return torch.sigmoid(self.deconv5(z))


class Vae(nn.Module):
    def __init__(self, dim_encoding):
        super().__init__()
        self.latent_space_vector = None
        self.z_dist = None
        self.encodings = None
        self.z = None
        self.dim_encoding = dim_encoding
        self.encoder = Encoder(dim_encoding)
        self.decoder = Decoder(dim_encoding)

    def reparameterize(self, encodings: Tensor) -> Tensor:
        mu = encodings[:, :self.dim_encoding]
        sigma = 1e-6 + F.softplus(encodings[:, self.dim_encoding:])

        z_dist = Normal(mu, sigma)
        self.z_dist = z_dist
        z = z_dist.rsample()
        return z

    def forward(self, x):
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
            beta=1.0,
            learning_rate=0.01,
            epochs=5
    ) -> tuple[nn.Module, list, list]:
        vl_fn = vae_loss_fn(beta)
        kl_div_fn = kl_loss()

        vae = self.to(device)
        optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate)
        training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        vae_loss_li = []
        kl_loss_li = []

        for epoch in range(epochs):
            i = 0
            for input, _ in training_dataloader:
                input = input.to(device)

                output = vae(input)

                # loss function to back-propagate on
                loss = vl_fn(input, output, vae.z_dist)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
                if i % 100 == 0:
                    # append vae loss
                    vae_loss_li.append(loss.item())

                    # calculate KL divergence loss
                    kl_loss_li.append(
                        kl_div_fn(vae.z_dist)
                    )
            print("Finished epoch: ", epoch + 1)
        return (
            self.to('cpu'),
            vae_loss_li,
            kl_loss_li
        )

    def generate_data(self, n_samples=32):
        device = next(self.parameters()).device
        _z = torch.randn(n_samples, 2).to(device)
        z = torch.randn(self.dim_encoding, device=device).repeat(n_samples, 1)
        z[:, 0:2] = _z
        with torch.no_grad():
            return self.decoder(z)


class ConditionalDecoder(nn.Module):
    def __init__(self, dim_encoding):
        super().__init__()
        self.dim_encoding = dim_encoding

        # z: (N, latent_dim + 10)
        self.fc1 = nn.Linear(in_features=dim_encoding + 10, out_features=448 * 2 * 2)
        self.bn1 = nn.BatchNorm1d(num_features=448 * 2 * 2)
        # z: (N, 448*2*2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=448, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        # z: (N, 256, 4, 4)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        # z: (N, 128, 8, 8)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        # z: (N, 64, 16, 16)

        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)
        # z: (N, 3, 32, 32)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         m.bias.data.fill_(0.)

    def forward(self, z):
        z = F.relu(self.bn1(self.fc1(z)))
        z = z.view(-1, 448, 2, 2)
        z = F.relu(self.bn2(self.deconv2(z)))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))
        return torch.sigmoid(self.deconv5(z))


class ConditionalVae(nn.Module):
    def __init__(self, dim_encoding):
        super().__init__()
        self.latent_space_vector = None
        self.z_dist = None
        self.encodings = None
        self.z = None
        self.dim_encoding = dim_encoding
        self.encoder = Encoder(dim_encoding)
        self.decoder = ConditionalDecoder(dim_encoding)

    def reparameterize(self, encodings: Tensor) -> Tensor:
        mu = encodings[:, :self.dim_encoding]
        sigma = 1e-6 + F.softplus(encodings[:, self.dim_encoding:])

        z_dist = Normal(mu, sigma)
        self.z_dist = z_dist
        z = z_dist.rsample()
        return z

    def forward(self, x, y):
        encodings = self.encoder(x)
        self.encodings = encodings
        z = self.reparameterize(encodings)

        assert z.shape[1] == self.dim_encoding
        self.latent_space_vector = z

        # Create one-hot vector from labels
        y = y.view(x.shape[0], 1)
        onehot_y = torch.zeros((x.shape[0], 10), device=device, requires_grad=False)
        onehot_y.scatter_(1, y, 1)

        latent = torch.cat((self.latent_space_vector, onehot_y), dim=1)
        return self.decoder(latent)

    def train_model(
            self,
            training_data,
            batch_size=64,
            beta=1.0,
            learning_rate=0.01,
            epochs=5
    ) -> tuple[nn.Module, list, list]:
        vl_fn = vae_loss_fn(beta)
        kl_div_fn = kl_loss()

        cvae = self.to(device)
        optimizer = torch.optim.Adam(params=cvae.parameters(), lr=learning_rate)
        training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        vae_loss_li = []
        kl_loss_li = []

        for epoch in range(epochs):
            i = 0
            for input, label in training_dataloader:
                input = input.to(device)
                label = label.to(device)

                output = cvae(input, label)

                # loss function to back-propagate on
                loss = vl_fn(input, output, cvae.z_dist)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
                if i % 100 == 0:
                    # append vae loss
                    vae_loss_li.append(loss.item())

                    # calculate KL divergence loss
                    kl_loss_li.append(
                        kl_div_fn(cvae.z_dist)
                    )
            print("Finished epoch: ", epoch + 1)
        return (
            self.to('cpu'),
            vae_loss_li,
            kl_loss_li
        )
    
    def generate_data(self, n_samples=32, target_label=0) -> tensor:
        """
        Generates random data samples (of size n) from the latent space
        """
        device = next(self.parameters()).device
        input_sample = torch.randn(n_samples, self.dim_encoding).to(device)

        assert input_sample.shape[0] == n_samples

        with torch.no_grad():
            label = torch.zeros((n_samples, 10), device=device)
            label[:, target_label] = 1
            latent = torch.cat((input_sample, label), dim=1)
            return self.decoder(latent)
