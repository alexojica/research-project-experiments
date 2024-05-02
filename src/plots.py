import matplotlib.pyplot as plt
from torch import Tensor
import numpy as np


def plot_latents(model, input, labels):
    z = model.encoder(input).detach().numpy()[:, :model.dim_encoding]
    plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', s=2.)
    plt.colorbar()
    plt.show()


def plot_image_data(images: np.ndarray, labels: np.ndarray):
    """
    Generate 5 subplots
    """
    plt.figure()
    for i in range(5):
        plt.subplot(151 + i)
        plt.axis('off')
        squeezed_img = np.squeeze(images[i])
        plt.imshow(squeezed_img)
        digit = np.argmax(labels[i])
        plt.title(digit)


def plot_training_result(
        input,
        labels,
        vae_model_classifier,
        total_losses,
        vae_loss_li,
        classifier_accuracy_li,
        classifier_loss_li,
        kl_loss_li
):
    plt.title('VAE classifier latent space')
    plot_latents(vae_model_classifier, input, labels)

    plt.plot(total_losses, label='VAE classifier -- total loss')
    plt.plot(vae_loss_li, label='VAE classifier -- vae loss')
    plt.legend()
    plt.show()

    plt.title('VAE classifier -- accuracy')
    plt.plot(classifier_accuracy_li)
    plt.show()

    plt.title('VAE classifier -- classifier loss')
    plt.plot(classifier_loss_li)
    plt.show()

    plt.title('VAE classifer -- KL divergence loss')
    plt.plot(kl_loss_li)
    plt.show()
