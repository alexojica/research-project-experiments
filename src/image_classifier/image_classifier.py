import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch import tensor
import numpy as np


class MNISTClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

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
        print('Finished Training')

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

    def test_model_syn_img(
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
