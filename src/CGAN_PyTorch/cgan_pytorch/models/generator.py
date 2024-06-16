# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

from CGAN_PyTorch.cgan_pytorch.utils.common import normal_init

model_urls = {
    "cgan": "https://github.com/Lornatang/CGAN-PyTorch/releases/download/v0.2.0/CGAN_MNIST-5fda105b1f24ad665b105873e9b8dcfc838bd892bce9373ac3035d109c61ed6e.pth"
}


class ConvGenerator(nn.Module):
    def __init__(self, image_size=28, channels=1, num_classes=10):
        super(ConvGenerator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(100 + num_classes, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_embedding(labels)), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Generator(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator.

    Args:
        image_size (int): The size of the image. (Default: 28)
        channels (int): The channels of the image. (Default: 1)
        num_classes (int): Number of classes for dataset. (Default: 10)
    """

    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10) -> None:
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(100 + num_classes, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(1024, channels * image_size * image_size),
            nn.Tanh()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, labels: list = None) -> torch.Tensor:
        """
        Args:
            inputs (tensor): input tensor into the calculation.
            labels (list):  input tensor label.

        Returns:
            A four-dimensional vector (N*C*H*W).
        """

        conditional_inputs = torch.cat([inputs, self.label_embedding(labels)], dim=-1)
        out = self.main(conditional_inputs)
        out = out.reshape(out.size(0), self.channels, self.image_size, self.image_size)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class GeneratorCIFAR(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super(GeneratorCIFAR, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8, momentum=0.9),
            nn.LeakyReLU(0.1),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, noise, condition):
        condition = condition.view(condition.size(0), -1)
        noise = noise.view(noise.size(0), -1)
        x = torch.cat((noise, condition), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 128, 8, 8)
        out = self.conv_layers(x)
        return out


def _gan(args):
    r""" Used to create GAN model.

    Args:
        arch (str): GAN model architecture name.
        image_size (int): The size of the image.
        channels (int): The channels of the image.
        pretrained (bool): If True, returns a model pre-trained on MNIST.
        progress (bool): If True, displays a progress bar of the download to stderr.

    Returns:
        Generator model.
    """
    if args.dataset == 'cifar':
        model = GeneratorCIFAR()
    else:
        model = Generator()

    return model


def cgan(args) -> Generator:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1406.2661>` paper.

    Args:
    """
    model = _gan(args)

    return model
