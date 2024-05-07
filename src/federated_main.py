#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from CGAN_PyTorch.cgan_pytorch.utils.common import configure
from CGAN_PyTorch.cgan_pytorch.models.discriminator import discriminator_for_mnist


def save_cgan_generated_images(global_model, device, num_classes=10, latent_dim=100):
    global_model.eval()  # Set the model to evaluation mode
    # Create a figure to display images
    fig, axs = plt.subplots(1, num_classes, figsize=(15, 3))

    # Generate one image per label
    for label in range(num_classes):
        z = torch.randn(1, latent_dim, device=device)  # Generate random noise
        labels = torch.LongTensor([label]).to(device)  # Create a tensor for the label

        with torch.no_grad():  # No need to track gradients
            generated_image = global_model(z, labels).detach().cpu()

        # Assuming output image is 1x28x28 (as in MNIST), adjust if different
        generated_image = generated_image.view(generated_image.size(1), generated_image.size(2), generated_image.size(3))  # Reshape image
        axs[label].imshow(generated_image.permute(1, 2, 0).squeeze(), cmap='gray' if generated_image.size(0) == 1 else None)
        axs[label].set_title(f'Label: {label}')
        axs[label].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'cgan':
        global_model = {'generator': configure("fed"), 'discriminator': discriminator_for_mnist(28, 1)}
        global_model['generator'].to(device)
        global_model['discriminator'].to(device)
        global_model['discriminator'].train()
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    if args.model != 'cgan':
        global_model.to(device)
        global_model.train()
    else:
        global_model['generator'].to(device)
        global_model['discriminator'].to(device)
        global_model['generator'].train()
        global_model['discriminator'].train()
    print(global_model)

    # copy weights
    if args.model == 'cgan':
        global_weights = {'generator': global_model['generator'].state_dict(), 'discriminator': global_model['discriminator'].state_dict()}
    else:
        global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        if args.model == 'cgan':
            global_model['discriminator'].train()
            global_model['generator'].train()
        else:
            global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights, args)

        # update global weights
        if args.model == 'cgan':
            global_model['generator'].load_state_dict(global_weights['generator'])
            global_model['discriminator'].load_state_dict(global_weights['discriminator'])
        else:
            global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        if args.model == 'cgan':
            global_model['discriminator'].eval()
            global_model['generator'].eval()
            save_cgan_generated_images(global_model['generator'], device)

        else:
            global_model.eval()

        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(global_model, args)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    if args.model == 'cgan':
        global_model['generator'].eval()
        save_cgan_generated_images(global_model['generator'], device)
        pass
    else:
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        # Saving the objects train_loss and train_accuracy:
        file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                   args.local_ep, args.local_bs)

        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
