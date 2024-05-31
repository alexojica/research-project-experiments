import os

import torch
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from torchvision import transforms

from train_cifar_classifier import CIFARResNetClassifier
from train_classifier_mnist import SimpleCNN
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

normalize = transforms.Normalize((0.5,), (0.5,))


def save_cgan_generated_images(global_model, device, dataset, num_classes=10, latent_dim=100):
    global_model.eval()  # Set the model to evaluation mode
    # Create a figure to display images
    fig, axs = plt.subplots(1, num_classes, figsize=(15, 3))

    # Generate one image per label
    for label in range(num_classes):
        z = torch.randn([1, latent_dim]).to(device) if dataset == 'mnist' else \
                        torch.randn(1, latent_dim, 1, 1, device=device)
        labels = torch.LongTensor([label]).to(device)  # Create a tensor for the label

        with torch.no_grad():  # No need to track gradients
            generated_image = global_model(z, labels).detach().cpu()
            # Manually scale the image data to [0, 1]
            generated_image = (generated_image - generated_image.min()) / (
                        generated_image.max() - generated_image.min())

        if dataset == 'mnist':
            # Assuming output image is 1x28x28 (as in MNIST), adjust if different
            generated_image = generated_image.view(generated_image.size(1), generated_image.size(2),
                                                   generated_image.size(3))  # Reshape image
        else:
            # Assuming output image is 3x32x32 (as in CIFAR), adjust if different
            generated_image = generated_image.view(generated_image.size(1), generated_image.size(2),
                                               generated_image.size(3))  # Reshape image
        axs[label].imshow(generated_image.permute(1, 2, 0).squeeze(),
                          cmap='gray' if generated_image.size(0) == 1 else None)
        axs[label].set_title(f'Label: {label}')
        axs[label].axis('off')

    plt.tight_layout()
    plt.show()


def calculate_emd(real_images, generated_images):
    # Move tensors to CPU and flatten them
    real_images = real_images.view(real_images.size(0), -1).cpu()
    generated_images = generated_images.view(generated_images.size(0), -1).cpu()
    # Calculate EMD
    emd = wasserstein_distance(real_images.mean(axis=0), generated_images.mean(axis=0))
    return emd


def classifier_accuracy(model, real_images, generated_images, labels, dataset, device='cuda'):
    if dataset == 'cifar':
        # Upsample images
        real_images = F.interpolate(real_images, size=(224, 224), mode='bilinear', align_corners=False).to(device)
        generated_images = F.interpolate(generated_images, size=(224, 224), mode='bilinear', align_corners=False).to(
            device)

    model.eval()
    with torch.no_grad():
        # Calculate accuracy for real and generated images
        if dataset == 'cifar':
            real_preds = model.model(real_images).argmax(dim=1) == labels
            fake_preds = model.model(generated_images).argmax(dim=1) == labels
        else:
            real_preds = model(real_images).argmax(dim=1) == labels
            fake_preds = model(generated_images).argmax(dim=1) == labels

        # Convert boolean tensor to float tensor before summing
        real_preds = real_preds.type(torch.float).sum().item()  # Convert to scalar
        fake_preds = fake_preds.type(torch.float).sum().item()  # Convert to scalar

        return real_preds, fake_preds


def generate_images(model, device, labels, num_images, latent_dim=100, dataset='mnist'):
    model.eval()
    z = torch.randn([num_images, latent_dim]).to(device) if dataset == 'mnist' else \
                        torch.randn(num_images, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        generated_images = model(z, labels).detach()
        generated_images = normalize(generated_images)
    return generated_images


def load_classifier(dataset, device='cuda', model_path=None):
    if model_path is None:
        model_path = os.path.join('weights', 'mnist_classifier.pth')
    if dataset == 'mnist':
        model = SimpleCNN()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model
    elif dataset == 'cifar':
        model = CIFARResNetClassifier()
        model.load_model()
        return model
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


def pca_images(real_images, fake_images, num_components=3):
    """real_images and fake_images are tensors of shape (batch_size, num_channels, height, width)"""
    # Flatten images
    real_images = real_images.view(real_images.size(0), -1).cpu()
    fake_images = fake_images.view(fake_images.size(0), -1).cpu()
    # Concatenate real and fake images
    images = torch.cat([real_images, fake_images], dim=0)
    # Standardize features by removing the mean and scaling to unit variance
    images = StandardScaler().fit_transform(images)
    # Apply PCA
    pca = PCA(n_components=num_components)
    pca.fit(images)
    # Project images to the principal components
    images_pca = pca.transform(images)
    # Split real and fake images
    real_images_pca = images_pca[:real_images.size(0)]
    fake_images_pca = images_pca[real_images.size(0):]
    return real_images_pca, fake_images_pca


def plot_pca(real_images_pca, fake_images_pca, labels, num_classes=10):
    """Plot PCA results of real and fake images.

    Args:
        real_images_pca (np.ndarray): PCA-transformed real images of shape (batch_size, num_components).
        fake_images_pca (np.ndarray): PCA-transformed fake images of shape (batch_size, num_components).
        labels (torch.Tensor): Tensor of shape (batch_size,) containing the class labels.
        num_classes (int): Number of classes.
    """
    labels = labels.cpu().numpy()
    plt.figure(figsize=(12, 8))

    # Define markers and colors
    markers = ['o', '^']
    colors = plt.cm.get_cmap('tab10', num_classes)

    # Plot real images
    for class_idx in range(num_classes):
        idx = labels == class_idx
        plt.scatter(real_images_pca[idx, 0], real_images_pca[idx, 1],
                    marker=markers[0], color=colors(class_idx),
                    label=f'Real Class {class_idx}', alpha=0.6)

    # Plot fake images
    for class_idx in range(num_classes):
        idx = labels == class_idx
        plt.scatter(fake_images_pca[idx, 0], fake_images_pca[idx, 1],
                    marker=markers[1], color=colors(class_idx),
                    label=f'Fake Class {class_idx}', alpha=0.6)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Real and Fake Images')
    plt.legend()
    plt.grid(True)
    plt.show()
