#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from sdv.metadata import SingleTableMetadata
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from baseline_main import save_model_with_timestamp
from gan_evaluation import plot_cgan_generated_images, load_classifier, generate_images, classifier_accuracy, \
    calculate_emd
from sdv_local.single_table.ctgan import CTGANSynthesizer, TVAESynthesizer
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, get_dataset_config, index_dataset, fed_avg, one_hot_encode
from CGAN_PyTorch.cgan_pytorch.utils.common import configure, weights_init
from CGAN_PyTorch.cgan_pytorch.models.discriminator import discriminator_for_mnist, get_discriminator


def testing(args, global_model, test_dataset, device, train_dataset=None, save_model=False, plot=False):
    if args.model != 'ctgan' and args.model != 'tvae':
        testloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    # Testing
    if args.model == 'cgan':
        # torch.save(global_model.state_dict(), os.path.join("weights", f"GAN-last.pth"))
        if plot:
            plot_cgan_generated_images(global_model, device, args.dataset, num_classes=args.num_classes,
                                       latent_dim=args.noise)
        real_preds = 0
        fake_preds = 0
        emd_real_images = torch.empty(0, 3, 32, 32).to(device)
        emd_fake_images = torch.empty(0, 3, 32, 32).to(device)
        labels_list = torch.empty(0, dtype=torch.long).to(device)

        model = load_classifier(args.dataset)
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)
            one_hot_labels = one_hot_encode(labels, num_classes=args.num_classes)

            fake_images = generate_images(global_model, device, one_hot_labels, num_images=images.size(0),
                                          latent_dim=args.noise,
                                          dataset=args.dataset)
            emd_real_images = torch.cat((emd_real_images, images), dim=0)
            emd_fake_images = torch.cat((emd_fake_images, fake_images), dim=0)
            labels_list = torch.cat((labels_list, labels), dim=0)
            real_preds_temp, fake_preds_temp = classifier_accuracy(model, images, fake_images, labels, args.dataset,
                                                                   device)
            real_preds += real_preds_temp
            fake_preds += fake_preds_temp

        real_accuracy = real_preds / len(test_dataset)
        fake_accuracy = fake_preds / len(test_dataset)

        emd = calculate_emd(emd_real_images, emd_fake_images)
        # pca_real, pca_fake = pca_images(emd_real_images, emd_fake_images)
        # plot_pca(pca_real, pca_fake, labels_list)
        print(f"Earth Mover's Distance: {emd}")
        print(f"Real images accuracy: {real_accuracy}, Fake images accuracy: {fake_accuracy}")

        if save_model:
            # Save model with timestamp
            save_model_with_timestamp(global_model, args=args)
        return fake_accuracy, emd
    elif args.model == 'ctgan' or args.model == 'tvae':
        global_model.save(os.path.join("weights", "ctgan_fed.pth"))
        # sample and save the dataframe to csv
        global_model._model._fitted = True
        global_model._fitted = True
        sample = global_model.sample(len(train_dataset))
        iid = 'iid' if args.iid else 'non_iid'
        sample.to_csv(os.path.join("synthetic_datasets", f"sample_fed_{args.model}_{args.dataset}_{iid}.csv"), index=False)
        return 0, 0
    else:
        test_acc, test_loss = test_inference(args, global_model['generator'], test_dataset)

        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        # Saving the objects train_loss and train_accuracy:
        file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                   args.local_ep, args.local_bs)

        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy], f)


def main(args):
    acc_list = []
    emds = []
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
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
        global_model = {'generator': configure(args), 'discriminator': get_discriminator(args.dataset, args.noise).to(device)}
        global_model['generator'].to(device)
        global_model['discriminator'].to(device)
        global_model['discriminator'].train()
        if args.dataset == 'cifar':
            global_model['generator'].apply(weights_init)
            global_model['discriminator'].apply(weights_init)
    elif args.model == 'ctgan' or args.model == 'tvae':
        if args.dataset == 'adult':
            config = get_dataset_config('adult_config.json')
            train_dataset = index_dataset(train_dataset, config)
            stm = SingleTableMetadata()
            has_id_column = config.pop("has_id")
            metadata = stm.load_from_dict(config)
        elif args.dataset == 'abalone':
            config = get_dataset_config('abalone_config.json')
            train_dataset = index_dataset(train_dataset, config)
            stm = SingleTableMetadata()
            has_id_column = config.pop("has_id")
            metadata = stm.load_from_dict(config)
        if args.model == 'ctgan':
            global_model = CTGANSynthesizer(metadata, train_dataset, cuda=True, epochs=30)
        else:
            global_model = TVAESynthesizer(metadata, train_dataset, cuda=True, epochs=30)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    if args.model == 'cgan':
        global_model['generator'].to(device)
        global_model['discriminator'].to(device)
        global_model['generator'].train()
        global_model['discriminator'].train()
    elif args.model != 'ctgan' and args.model != 'tvae':
        global_model.to(device)
        global_model.train()
    print(global_model)

    # copy weights
    if args.model == 'cgan':
        global_weights = {'generator': global_model['generator'].state_dict(), 'discriminator': global_model['discriminator'].state_dict()}
    elif args.model == 'ctgan' or args.model == 'tvae':
        global_weights = global_model.get_weights()
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
        elif args.model != 'ctgan' and args.model != 'tvae':
            global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # idxs_users = range(args.num_users)

        dataset_size_per_client = [len(user_groups[i]) for i in idxs_users]

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            if args.model == 'ctgan' or args.model == 'tvae':
                w, loss, data_processor = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
            else:
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights, args)
        # global_weights = fed_avg(local_weights, dataset_size_per_client, args)

        # update global weights
        if args.model == 'cgan':
            global_model['generator'].load_state_dict(global_weights['generator'])
            global_model['discriminator'].load_state_dict(global_weights['discriminator'])
        elif args.model == 'ctgan' or args.model == 'tvae':
            global_model.set_weights(global_weights)
            global_model._data_processor = data_processor
        else:
            global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        if args.model != 'ctgan' and args.model != 'tvae':
            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            if args.model == 'cgan':
                # global_model['discriminator'].eval()
                # global_model['generator'].eval()
                # save_cgan_generated_images(global_model['generator'], device, args.dataset, latent_dim=args.noise)
                pass
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
            if (epoch+1) % args.testevery == 0:
                plot = True if (epoch + 1) % 20 == 0 or (epoch + 1) == 5 else False
                acc, emd = testing(args, global_model['generator'], test_dataset, device, train_dataset, plot=plot)
                acc_list.append(acc)
                emds.append(emd)
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    testing(args, global_model, test_dataset, device, train_dataset, plot=True)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    return acc_list, emds

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


if __name__ == '__main__':
    # tvae iid
    args = args_parser()
    main(args)

    # tvae non-iid
    args.iid = 0
    main(args)

    # ctgan iid
    args.model = 'ctgan'
    args.iid = 1
    main(args)

    # ctgan non-iid
    args.iid = 0
    main(args)

