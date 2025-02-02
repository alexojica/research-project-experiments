#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import pandas as pd


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def split_dirichlet(dataset, num_users: int, is_cfar: bool, beta: float = 0.5) -> dict[int, [int]]:
    """
    Sample non-I.I.D client data from an arbitary dataset.
    Samples it based on this paper: 10.48550/ARXIV.1905.12022
    :param dataset: The dataset
    :param num_users: The number of clients
    :param is_cfar: Whether the dataset is cfar (bool). This is a stupid solution but is necessary since for some reason,
    their parameter name is different.
    :param beta: The beta parameter used to control the distribution spread.
    :return: dict mapping client id to idxs for training
    """
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets) if is_cfar else dataset.train_labels.numpy()
    uniq_labels = np.unique(labels)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs_labels = idxs_labels.T

    assert np.shape(idxs_labels) == (len(dataset), 2)

    for label in uniq_labels:
        relevant_idxs = idxs_labels[(idxs_labels[:, 1] == label)][:, 0].T
        proportions = np.random.dirichlet(np.full(num_users, beta))
        splits = split_by_ratio(relevant_idxs, proportions)
        for idx, split in enumerate(splits):
            dict_users[idx] = np.concatenate([dict_users[idx], split])

    for _, dict_val in dict_users.items():
        if len(dict_val) < 40:
            # We just restart a split if a user isn't assigned enough samples.
            return split_dirichlet(dataset, num_users, is_cfar, beta)

    return dict_users


def split_by_ratio(arr, ratios):
    """
    Splits an np array according to some proportions, must sum to 1
    """
    arr = np.random.permutation(arr)
    ind = np.add.accumulate(np.array(ratios) * len(arr)).astype(int)
    return [x.tolist() for x in np.split(arr, ind)][:len(ratios)]


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def adult_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def adult_noniid(dataset, num_users):
    # Shuffle the dataframe to ensure randomness
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Group the data by 'education' level
    grouped_data = dataset.groupby('education')

    # Create a list of data indices for each education level
    grouped_indices = [group.index.tolist() for name, group in grouped_data]

    # Flatten the list of grouped indices
    all_indices = [idx for sublist in grouped_indices for idx in sublist]

    # Calculate the number of samples each user should get
    samples_per_user = len(all_indices) // num_users

    # Create a dictionary to store the non-IID split indices
    user_indices = {i: [] for i in range(num_users)}

    # Distribute indices to users in a non-IID fashion
    for i, idx in enumerate(all_indices):
        user_idx = i % num_users
        user_indices[user_idx].append(idx)

    # Ensure each user gets the same number of samples
    for user in range(num_users):
        user_indices[user] = user_indices[user][:samples_per_user]

    return user_indices


def adult_noniid_unequal(dataset, num_users):
    num_shards = 50  # More shards for more granularity
    num_items_per_shard = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: pd.DataFrame() for i in range(num_users)}
    idxs = np.arange(len(dataset))

    random_shard_size = np.random.randint(1, 10, size=num_users)  # Randomly choose between 1 and 10 shards for each user
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards).astype(int)

    for i in range(num_users):
        shard_size = random_shard_size[i]
        rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = pd.concat(
                (dict_users[i], dataset.iloc[shards[rand]]))
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
