import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch import tensor


MNIST_INPUT_SIZE = 784
MNIST_HIDDEN_LAYER_SIZE_1 = 512
MNIST_HIDDEN_LAYER_SIZE_2 = 256
MNIST_NUM_CLASSES = 10


class MNISTClassifier(nn.Module):
    def __init__(self, input_size=MNIST_INPUT_SIZE, num_classes=MNIST_NUM_CLASSES):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, MNIST_HIDDEN_LAYER_SIZE_1)
        self.fc2 = nn.Linear(MNIST_HIDDEN_LAYER_SIZE_1, MNIST_HIDDEN_LAYER_SIZE_2)
        self.fc3 = nn.Linear(MNIST_HIDDEN_LAYER_SIZE_2, num_classes)

    def forward(self, x: tensor):
        """
        x should have shape [_, 1, 28 ,28]

        This will be flattened to [_, 784]
        """
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def train_model(
            self,
            training_data,
            batch_size,
            epochs
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters())
        training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for input, labels in training_dataloader:
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(input)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print("Epoch done: ", epoch + 1)

    def test_model(
            self,
            testing_data: Dataset
    ) -> float:
        test_dataloader = DataLoader(testing_data, shuffle=True)

        correct = 0
        total = 0
        for input, labels in test_dataloader:
            outputs = self.forward(input)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total

    def generate_labels(
            self,
            input: tensor
    ) -> tensor:
        labels = []
        for img in input:
            outputs = self.forward(img)
            _, predicted = torch.max(outputs.data, 1)
            labels.append(predicted)
        return torch.stack(labels)

    def test_model_syn_img_label(
            self,
            testing_data: tensor,
            labels: tensor
    ) -> float:
        correct = 0
        total = 0
        for i, input in enumerate(testing_data):
            outputs = self.forward(input)
            _, predicted = torch.max(outputs.data, 1)
            _, label = torch.max(labels[i], 0)
            total += 1
            correct += (predicted == label).sum().item()
        return correct / total


CIFAR_INPUT_SIZE = 784
CIFAR_HIDDEN_LAYER_SIZE_1 = 512
CIFAR_HIDDEN_LAYER_SIZE_2 = 256
CIFAR_NUM_CLASSES = 10
KERNEL_SIZE = 4
STRIDE_LENGTH = 2
PADDING_SIZE = 1

train_on_gpu = torch.cuda.is_available()


class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x

    def train_model(
            self,
            training_data,
            batch_size,
            learning_rate,
            epochs
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            for input, labels in training_dataloader:
                if train_on_gpu:
                    input, labels = input.cuda(), labels.cuda()
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(input)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print("Epoch done: ", epoch + 1)

    def test_model(
            self,
            testing_data: Dataset
    ) -> float:
        test_dataloader = DataLoader(testing_data, shuffle=True)

        correct = 0
        total = 0
        for input, labels in test_dataloader:
            if train_on_gpu:
                input, labels = input.cuda(), labels.cuda()
            output = self.forward(input)
            _, pred = torch.max(output, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return correct / total

    def generate_labels(
            self,
            input: tensor
    ) -> tensor:
        labels = []
        outputs = self.forward(input)
        for output in outputs:
            labels.append(torch.argmax(output).cpu().item())
        return labels

    def test_model_syn_img_label(
            self,
            testing_data: tensor,
            labels: tensor
    ) -> float:
        correct = 0
        total = 0
        for i, input in enumerate(testing_data):
            outputs = self.forward(input)
            _, predicted = torch.max(outputs.data, 1)
            _, label = torch.max(labels[i], 0)
            total += 1
            correct += (predicted == label).sum().item()
        return correct / total