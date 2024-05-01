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


def vae_classifier_loss_fn(
        input,
        output,
        model_encodings,
        labels,
        alpha=1.0
) -> float:
    """
    Loss function for the VAE with classification in output. It considers three terms:
    - reconstruction loss by comparing image quality between input and output
    - difference between the current and desired latent probability distribution, computed with
    Kullback-Leibler divergence (KL)
    - Cross-entropy loss to minimize error between the actual and predicted outcomes
    """

    # regular loss function
    mse = nn.MSELoss(reduction='sum')
    reg_loss = mse(input, output)

    # KL loss function
    mu, sigma = get_properties(model_encodings)
    kl_loss = 0.5 * torch.sum(sigma ** 2 + mu ** 2 - 1 - 2 * torch.log(sigma))

    # cross-entropy loss function
    cl = nn.CrossEntropyLoss()
    cl_loss = alpha * cl(output[1], labels)

    return reg_loss + kl_loss + cl_loss
