#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import Tensor
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from torch import nn
from torch.utils.data import DataLoader


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


# TODO: get this verified
def get_properties(encodings: Tensor, dim_encoding: int) -> tuple[Tensor, Tensor]:
    """
    Extracts the mean (mu) and standard deviation (sd) of the encodings
    """
    mu = encodings[:, :dim_encoding]
    sigma = torch.exp(encodings[:, dim_encoding:])
    return mu, sigma


# TODO: get this verified
def sample(encodings: Tensor, dim_encoding: int) -> Tensor:
    """
    Given encodings and its dimensionality, generates samples its distribution.

    For example: if given 6 data points with 2-dimensional encoding:
    - encodings: torch.Size([6, 2])
    - returns: torch.Size([6, 2])
    """
    mu, sigma = get_properties(encodings, dim_encoding)
    s = torch.rand(mu.shape, device=encodings.device)
    return mu + s * torch.sqrt(sigma)


def reg_loss_fn():
    mse = nn.MSELoss(reduction='sum')
    return lambda input, output: mse(input, output)


# TODO: get this verified
def kl_loss(model_encodings, dim_encoding):
    mu, sigma = get_properties(model_encodings, dim_encoding)
    return 0.5 * torch.sum(sigma**2 + mu**2 - 1 - 2*torch.log(sigma))


def vae_loss_fn():
    reg = reg_loss_fn()
    return lambda input, output, model_encodings, dim_encoding:\
        reg(input, output) +\
        kl_loss(model_encodings, dim_encoding)


def vae_classifier_loss_fn(alpha):
    """
    Loss function for the VAE with classification to use for backpropagation. It considers three terms:
    - reconstruction loss by comparing image quality between input and output
    - difference between the current and desired latent probability distribution, computed with
    Kullback-Leibler divergence (KL)
    - Cross-entropy loss to minimize error between the actual and predicted outcomes
    """
    vl_fn = vae_loss_fn()
    cl_fn = nn.CrossEntropyLoss()

    return lambda input, output, model_encodings, dim_encoding, labels: \
        vl_fn(input, output[0], model_encodings, dim_encoding) + \
        alpha * cl_fn(output[1], labels)


def train_vae_classifier(
        vae_classifier_model: nn.Module,
        training_data,
        alpha=1.0,
        epochs=5
) -> tuple[nn.Module, list, list, list, list, list]:
    complete_loss_fn = vae_classifier_loss_fn(alpha)
    cl_fn = nn.CrossEntropyLoss()
    vl_fn = vae_loss_fn()

    vae_classifier_model = vae_classifier_model.to('cuda')
    optimizer = torch.optim.Adam(params=vae_classifier_model.parameters())

    training_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    total_losses = []
    classifier_accuracy_li = []
    classifier_loss_li = []
    vae_loss_li = []
    kl_loss_li = []

    for epoch in range(epochs):
        i = 0
        for input, labels in training_dataloader:
            input = input.to('cuda')
            labels = labels.to('cuda')
            output = vae_classifier_model(input)

            # loss function to back-propagate on
            loss = complete_loss_fn(
                input,
                output,
                vae_classifier_model.encodings,
                vae_classifier_model.dim_encoding,
                labels
            )

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            if i % 100 == 0:
                total_losses.append(loss.item())

                # calculate accuracy
                matches_labels = (torch.argmax(output[1], 1) == labels)
                accuracy = torch.mean(matches_labels.float())
                classifier_accuracy_li.append(accuracy)

                # calculate cross entropy loss
                classifier_loss_li.append(
                    cl_fn(output[1], labels)
                )

                # calculate VAE loss
                vae_loss_li.append(
                    vl_fn(input, output[0], vae_classifier_model.encodings, vae_classifier_model.dim_encoding)
                )

                # calculate KL loss
                kl_loss_li.append(
                    kl_loss(vae_classifier_model.encodings, vae_classifier_model.dim_encoding)
                )

                print(epoch, loss.item(), end='; ')
    return (
        vae_classifier_model.to('cpu'),
        total_losses,
        classifier_accuracy_li,
        classifier_loss_li,
        vae_loss_li,
        kl_loss_li
    )
