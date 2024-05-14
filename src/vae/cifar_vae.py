import torch
from torch import device, cuda, tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = device('cuda' if cuda.is_available() else 'cpu')


class Encoder(nn.Module):

    def __init__(self, dim_encoding):
        super().__init__()
        self.dim_encoding = dim_encoding

        # x: (N, 3, 32, 32+10)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        # x: (N, 64, 16, 21)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        # x: (N, 128, 8, 10)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        # x: (N, 256, 4, 5)
        self.fc4 = nn.Linear(in_features=256*4*5, out_features=2*dim_encoding)
        # x: (N, 2*latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.)

    def forward(self, x):
        """
        mu: Vector of size latent_dim.
            Each element represents the mean of a Gaussian Distribution.
        sigma: Vector of size latent_dim.
               Each element represents the standard deviation of a Gaussian distribution.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = x.view(-1, 256*4*5)
        x = self.fc4(x)

        # split x in half
        mu = x[:, :self.dim_encoding]
        # sigma shouldn't be negative
        sigma = 1e-6 + F.softplus(x[:, self.dim_encoding:])

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, dim_encoding):
        """
        Parameter:
            dim_encoding: Dimension of the latent variable
            model_sigma: Whether to model standard deviations too.
                         If False, only outputs the mu vector, and all sigma is implicitly 1.
        """
        super().__init__()
        self.dim_encoding = dim_encoding

        # z: (N, latent_dim+10)
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

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.)

    def forward(self, z):
        """
            mu: Vector of size latent_dim.
                Each element represents the mean of a Gaussian Distribution.
            sigma: Vector of size latent_dim.
                   Each element represents the standard deviation of a Gaussian distribution.
        """
        z = F.relu(self.bn1(self.fc1(z)))
        z = z.view(-1, 448, 2, 2)
        z = F.relu(self.bn2(self.deconv2(z)))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))
        mu = torch.sigmoid(self.deconv5(z))
        return mu


class ConditionalVae(nn.Module):
    def __init__(self, dim_encoding):
        super().__init__()
        self.z = None
        self.dim_encoding = dim_encoding
        self.encoder = Encoder(dim_encoding)
        self.decoder = Decoder(dim_encoding)

    def forward(self, x, y):
        # Create one-hot vector from labels
        y = y.view(x.shape[0], 1)
        onehot_y = torch.zeros((x.shape[0], 10), device=device, requires_grad=False)
        onehot_y.scatter_(1, y, 1)

        # Encode
        onehot_conv_y = onehot_y.view(x.shape[0], 1, 1, 10) * torch.ones((x.shape[0], x.shape[1], x.shape[2], 10),
                                                                         device=device)
        input_batch = torch.cat((x, onehot_conv_y), dim=3)
        z_mu, z_sigma = self.encoder(input_batch)

        # reparametrization trick
        self.z = z_mu + z_sigma * torch.randn_like(z_mu, device=device)

        # Decode
        latent = torch.cat((self.z, onehot_y), dim=1)
        output = self.decoder(latent)
        return z_mu, z_sigma, output

    def train_model(
            self,
            training_data,
            batch_size=64,
            beta=1.0,
            learning_rate=0.01,
            epochs=5
    ) -> tuple[nn.Module, list, list]:
        """
        https://github.com/jaywonchung/Learning-ML/tree/master/Implementations/Conditional-Variational-Autoencoder
        """
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

                z_mu, z_sigma, output = cvae(input, label)

                kl_loss = 0.5 * torch.sum(z_mu ** 2 + z_sigma ** 2 - torch.log(1e-8 + z_sigma ** 2) - 1., dim=1)
                reconstruction_loss = -0.5 * torch.sum((input - output) ** 2, dim=(1, 2, 3))

                ELBO_i = reconstruction_loss - kl_loss
                loss = -torch.mean(ELBO_i)

                # # loss function to back-propagate on
                # loss = kl_loss + reconstruction_loss

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
                if i % 100 == 0:
                    vae_loss_li.append(loss.item())
                    kl_loss_li.append(reconstruction_loss)
            print("Finished epoch: ", epoch + 1)
        return (
            self.to('cpu'),
            vae_loss_li,
            kl_loss_li
        )
    
    def generate_data(self, n_samples=32, target_label=1) -> tensor:
        """
        Generates random data samples (of size n) from the latent space
        """
        device = next(self.parameters()).device
        input_sample = torch.randn(n_samples, self.dim_encoding).to(device)

        assert input_sample.shape[0] == n_samples

        # # Generate output from decoder
        # i = 1
        # label = torch.zeros((n_samples, 10), device=device)
        # label[:, i] = 1
        # latent = torch.cat((input_sample, label), dim=1)
        # output = self.decoder(latent)
        # return output
        with torch.no_grad():
            label = torch.zeros((n_samples, 10), device=device)
            print(label)
            label[:, target_label] = 1
            print(label)
            latent = torch.cat((input_sample, label), dim=1)
            return self.decoder(latent)
