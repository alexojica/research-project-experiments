import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import os

from utils import get_dataset


class CIFARResNetClassifier:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = models.resnet50(pretrained=True)
        self._prepare_model()

    def _prepare_model(self):
        # Freeze all layers in the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer with a new classifier
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )
        self.model.to(self.device)

    def train(self, trainset, testset, epochs=200, batch_size=64, lr=0.00002):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=lr)
        epoch_losses = []
        epoch_accuracies = []
        test_accuracies = []

        for epoch in tqdm(range(epochs), "Overall progress: "):
            batch_losses = []
            correct = 0
            total = 0
            for inputs, labels in DataLoader(trainset, batch_size=batch_size, shuffle=True):
                inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False).to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_accuracy = 100. * correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            test_loss, test_accuracy = self.test(testset, batch_size)
            test_accuracies.append(test_accuracy)

            if (epoch + 1) % 50 == 0:
                self.save_model(f'weights/cifar_classifier_epoch_{epoch+1}.pth')

        return epoch_losses, epoch_accuracies, test_accuracies

    def test(self, testset, batch_size=64):
        self.model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in DataLoader(testset, batch_size=batch_size, shuffle=False):
                inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                test_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(testset)
        accuracy = 100. * correct / len(testset)
        return test_loss, accuracy

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join('weights', 'cifar_classifier.pth'))

    def load_model(self, path=None):
        if path is None:
            path = os.path.join('weights', 'cifar_classifier.pth')
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def eval(self):
        self.model.eval()

    def predict(self, inputs):
        return self.model(inputs)


if __name__ == '__main__':
    classifier = CIFARResNetClassifier()
    classifier.load_model()
    trainset, testset, _ = get_dataset('cifar')
    test_loss, acc = classifier.test(testset)
    print(f'Test accuracy: {acc:.2f}%')

