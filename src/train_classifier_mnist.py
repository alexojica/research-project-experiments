import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_dataset_custom_training
import matplotlib.pyplot as plt
import os


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = torch.nn.Linear(1024, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_classifier_mnist(model, trainset, testset, device='cuda'):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    epoch_losses = []
    epoch_accuracies = []
    test_accuracies = []

    for epoch in tqdm(range(2), "Overall progress: "):
        model.train()
        batch_losses = []
        correct = 0
        total = 0
        for inputs, labels in tqdm(DataLoader(trainset, batch_size=16, shuffle=True), f"Epoch {epoch + 1}:"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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

        model.eval()
        test_loss, test_accuracy = test_model(model, testset, device)
        test_accuracies.append(test_accuracy)

    return model, epoch_losses, epoch_accuracies, test_accuracies


def test_model(model, testset, device='cuda'):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in DataLoader(testset, batch_size=64, shuffle=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(testset)
    accuracy = 100. * correct / len(testset)
    return test_loss, accuracy


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use get_dataset to load MNIST data
    train_dataset, test_dataset = get_dataset_custom_training('mnist')

    model = SimpleCNN()
    trained_model, train_losses, train_accuracies, test_accuracies = train_classifier_mnist(model, train_dataset,
                                                                                            test_dataset, device)
    
    # Plotting test accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the final test accuracy
    final_test_accuracy = test_accuracies[-1]
    print(f'Final Test Accuracy: {final_test_accuracy:.2f}%')

    torch.save(trained_model.state_dict(), os.path.join('weights', 'mnist_classifier_new.pth'))
    print(f'Model saved to' + os.path.join('weights', 'mnist_classifier_new.pth'))
