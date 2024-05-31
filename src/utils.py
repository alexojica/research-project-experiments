#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import json
import os

import pandas as pd
import torch
from sampling import *


def index_dataset(input_dataset: pd.DataFrame, input_configuration) -> pd.DataFrame:
    """Ensures the index column is always in the dataset, as the first column

    Args:
        input_configuration: check for existence of index column
        input_dataset: the dataset whose columns are being checked/altered

    Returns:
       the new dataset with the index column
    """
    if input_configuration['has_id'] == "False":
        input_dataset['index'] = input_dataset.index
        columns = input_dataset.columns.tolist()
        # Move the last column to the first position
        columns = [columns[-1]] + columns[:-1]
        # Reorder the DataFrame columns
        input_dataset = input_dataset[columns]
    else:
        input_dataset.reset_index(inplace=True)
        input_dataset['index'] = input_dataset.index
    return input_dataset


def get_path_to_file(file_name, folder_name):
    """Finds the path from the root folder to the specified file

    Args:
        file_name: The name of the file
        folder_name: The name of the folder the file is in

    Returns:
        The absolute path to the file
    """

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the project's root directory
    return os.path.join(project_root, folder_name, file_name)


def get_input_dataset(input_dataset: str):
    """Validates the given parameter for path to input dataset

    Args:
        input_dataset: string path to the location/name of the input dataset

    Returns:
        a pandas dataframe holding the input dataset

    Raises:
        ValueError: If the input dataset is not specified
        FileNotFoundError: If the input dataset is not found
    """
    if input_dataset is None:
        raise ValueError("No input dataset specified. Please provide a path to the dataset. Ending program...\n")

    try:
        path_to_dataset = get_path_to_file(input_dataset, "data")
        dataset = pd.read_csv(path_to_dataset)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not find dataset - \"{input_dataset}\". Ending program...\n") from exc

    return dataset


def get_dataset_config(input_configuration: str):
    """Validates the given parameter for path to dataset configuration

    Args:
        input_configuration: string path to the location/name of the dataset configuration

    Returns:
        a dictionary holding the dataset configuration

    Raises:
        ValueError: If the input configuration is not specified
        FileNotFoundError: If the input configuration is not found
    """
    if input_configuration is None:
        raise ValueError("No configuration file specified for the dataset. "
                         "Please provide a path to the configuration file for the given dataset. Ending program...\n")

    try:
        path_to_config = get_path_to_file(input_configuration, "datasets_configs")
        config = json.load(open(path_to_config, 'r', encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not find config file - \"{input_configuration}\". Ending program...\n") from exc

    return config


def get_dataset_custom_training(dataset):
    if dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

    return train_dataset, test_dataset


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
        if args.iid == 1:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        elif args.iid == 2:
            user_groups = split_dirichlet(train_dataset, args.num_users, is_cfar=True, beta=args.dirichlet)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)
    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid == 1:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        elif args.iid == 2:
            user_groups = split_dirichlet(train_dataset, args.num_users, is_cfar=False, beta=args.dirichlet)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    elif args.dataset == 'adult':
        train_dataset = get_input_dataset('adult.csv')
        test_dataset = None
        if args.iid == 1:
            # Sample IID user data from Mnist
            user_groups = adult_iid(train_dataset, args.num_users)
        # elif args.iid == 2:
        #     user_groups = split_dirichlet(train_dataset, args.num_users, is_cfar=False, beta=args.dirichlet)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = adult_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = adult_noniid(train_dataset, args.num_users)
    return train_dataset, test_dataset, user_groups


def average_weights(w, args):
    """
    Returns the average of the weights.
    """
    if args.model == 'cgan' or args.model == 'ctgan':
        w_avg_g = copy.deepcopy(w[0]['generator'])
        w_avg_d = copy.deepcopy(w[0]['discriminator'])
        for key in w_avg_g.keys():
            for i in range(1, len(w)):
                w_avg_g[key] += w[i]['generator'][key]
            w_avg_g[key] = torch.div(w_avg_g[key], len(w))
        for key in w_avg_d.keys():
            for i in range(1, len(w)):
                w_avg_d[key] += w[i]['discriminator'][key]
            w_avg_d[key] = torch.div(w_avg_d[key], len(w))
        return {'generator': w_avg_g, 'discriminator': w_avg_d}
    elif args.model == 'tvae':
        w_avg_e = copy.deepcopy(w[0]['encoder'])
        w_avg_d = copy.deepcopy(w[0]['decoder'])
        for key in w_avg_e.keys():
            for i in range(1, len(w)):
                w_avg_e[key] += w[i]['encoder'][key]
            w_avg_e[key] = torch.div(w_avg_e[key], len(w))
        for key in w_avg_d.keys():
            for i in range(1, len(w)):
                w_avg_d[key] += w[i]['decoder'][key]
            w_avg_d[key] = torch.div(w_avg_d[key], len(w))
        return {'encoder': w_avg_e, 'decoder': w_avg_d}
    else:
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
