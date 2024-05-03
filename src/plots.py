import matplotlib.pyplot as plt
import numpy as np


def plot_two_d_latents(model, input, labels):
    input_encodings = model.encoder(input).detach().numpy()[:, :model.dim_encoding]
    plt.title('VAE classifier latent space')
    plt.scatter(input_encodings[:, 0], input_encodings[:, 1], c=labels, cmap='tab10', s=2.)
    plt.colorbar()
    plt.show()


def plot_three_d_latents(model, input, labels):
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")
    input_encodings = model.encoder(input).detach().numpy()[:, :model.dim_encoding]
    sctt = ax.scatter3D(input_encodings[:, 0], input_encodings[:, 1], input_encodings[:, 2],
                        alpha=0.8,
                        c=labels,
                        cmap='tab10')
    plt.title("VAE classifier latent space")
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
    plt.show()


def plot_image_data(images: np.ndarray, label_probabilities: np.ndarray):
    """
    Generate 5 subplots
    """
    plt.figure()
    for i in range(5):
        plt.subplot(151 + i)
        plt.axis('off')
        squeezed_img = np.squeeze(images[i])
        plt.imshow(squeezed_img)
        digit = np.argmax(label_probabilities[i])
        plt.title(digit)


def plot_image_data_two(images: np.ndarray, labels: np.ndarray):
    plt.figure()
    for i in range(5):
        plt.subplot(151 + i)
        plt.axis('off')
        squeezed_img = np.squeeze(images[i])
        plt.imshow(squeezed_img)
        plt.title(labels[i])


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
    if vae_model_classifier.dim_encoding == 2:
        plot_two_d_latents(vae_model_classifier, input, labels)
    elif vae_model_classifier.dim_encoding == 3:
        plot_three_d_latents(vae_model_classifier, input, labels)

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
