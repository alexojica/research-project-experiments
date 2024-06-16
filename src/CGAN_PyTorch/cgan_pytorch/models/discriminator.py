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
import torch.nn.functional as F

from CGAN_PyTorch.cgan_pytorch.utils.common import normal_init


class DiscriminatorForMNIST(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network discriminator.

    Args:
        image_size (int): The size of the image. (Default: 28)
        channels (int): The channels of the image. (Default: 1)
        num_classes (int): Number of classes for dataset. (Default: 10)
    """

    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10) -> None:
        super(DiscriminatorForMNIST, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(channels * image_size * image_size + num_classes, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, labels: list = None) -> torch.Tensor:
        r""" Defines the computation performed at every call.

        Args:
            inputs (tensor): input tensor into the calculation.
            labels (list):  input tensor label.

        Returns:
            A four-dimensional vector (N*C*H*W).
        """
        inputs = torch.flatten(inputs, 1)
        conditional = self.label_embedding(labels)
        conditional_inputs = torch.cat([inputs, conditional], dim=-1)
        out = self.main(conditional_inputs)

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


def discriminator_for_mnist(image_size: int = 28, channels: int = 1) -> DiscriminatorForMNIST:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1406.2661>` paper.
    """
    model = DiscriminatorForMNIST(image_size, channels)

    return model


class DiscriminatorCIFAR(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super(DiscriminatorCIFAR, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1)
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(128 * (input_shape[1] // 8) * (input_shape[2] // 8) + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input, condition):
        x = self.conv_layers(input)
        x = self.flatten(x)
        condition = condition.view(condition.size(0), -1)  # Ensure condition is flattened
        x = torch.cat((x, condition), dim=1)
        out = self.fc(x)
        return out


def get_discriminator(dataset: str, noise_size=100) -> nn.Module:
    if dataset == 'mnist':
        return DiscriminatorForMNIST()
    elif dataset == 'cifar':
        return DiscriminatorCIFAR()
    else:
        raise ValueError("Unsupported dataset")
