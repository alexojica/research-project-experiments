import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models

from utils import get_dataset
import os


def train_classifier_cifar(model, trainset, testset, device='cuda'):
    # Freeze all layers in the model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer with a new classifier
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(num_ftrs, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
        torch.nn.Softmax(dim=1)
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.00002)

    # Train the model on your train dataset
    epoch_losses = []
    epoch_accuracies = []
    test_accuracies = []
    
    for epoch in tqdm(range(200), "Overall progress: "):
        model.train()
        batch_losses = []
        correct = 0
        total = 0
        for inputs, labels in tqdm(DataLoader(trainset, batch_size=64, shuffle=True), f"Epoch {epoch + 1}:"):
            inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False).to(device)
            # print(inputs.shape)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate batch loss and accuracy
            batch_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_accuracy = 100. * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        
        print(f'Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%')
        with torch.no_grad():
            model.eval()
            # Test the model after each training epoch
            test_loss, test_accuracy = test_model(model, testset, device)
            print(f'Epoch {epoch + 1} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
            test_accuracies.append(test_accuracy)
        
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join('weights', f'cifar_classifier_epoch_{epoch+1}.pth'))
            print(f'Model saved at epoch {epoch+1}')
    
    return model, epoch_losses, epoch_accuracies, test_accuracies


def test_model(model, testset, device='cuda'):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in DataLoader(testset, batch_size=64, shuffle=False):
            # Upsample images to 224x224
            inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(testset)
    accuracy = 100. * correct / len(testset)
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the pretrained ResNet50 model
    resnet50_model = models.resnet50(pretrained=True).to(device)

    train_dataset, test_dataset, _ = get_dataset('cifar')

    # Train the classifier
    trained_model, train_losses, train_accuracies, test_accuracies = train_classifier_cifar(resnet50_model, train_dataset, test_dataset, device)
    test_loss, test_accuracy = test_model(trained_model, test_dataset, device)
    
    # Plotting training loss and accuracy
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.title('Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.show()
