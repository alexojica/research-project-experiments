#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from torch.nn import BCELoss
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import os
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
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
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)
    batch_size = 64
    discriminator = None
    adversarial_loss = None

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
        if args.dataset == 'mnist':
            global_model = configure("fed")
            discriminator = discriminator_for_mnist(28, 1)
            discriminator.to(device)
            discriminator.train()

            # Set optimizers for both Generator and Discriminator
            optimizer_G = Adam(global_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
            optimizer_D = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

            # Set loss function
            adversarial_loss = torch.nn.MSELoss().to(device)

            fixed_noise = torch.randn([batch_size, 100]).to(device)
            fixed_conditional = torch.randint(0, 1, (batch_size,)).to(device)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs), "Progress overall"):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(tqdm(trainloader, f"Epoch {epoch + 1}")):
            images = images.to(device)
            labels = labels.to(device)
            if args.model == 'cgan':
                discriminator.train()
                global_model.train()
                btch_size = images.size(0)
                real_label = torch.full((btch_size, 1), 1, dtype=images.dtype).to(device)
                fake_label = torch.full((btch_size, 1), 0, dtype=images.dtype).to(device)

                noise = torch.randn([btch_size, 100]).to(device)
                conditional = torch.randint(0, 10, (btch_size,)).to(device)

                ##############################################
                # (1) Update D network: max E(x)[log(D(x))] + E(z)[log(1- D(z))]
                ##############################################
                # Set discriminator gradients to zero.
                discriminator.zero_grad()

                # Train with real.
                real_output = discriminator(images, labels)
                d_loss_real = adversarial_loss(real_output, real_label)
                d_loss_real.backward()
                d_x = real_output.mean()

                # Train with fake.
                fake = global_model(noise, conditional)
                fake_output = discriminator(fake.detach(), conditional)
                d_loss_fake = adversarial_loss(fake_output, fake_label)
                d_loss_fake.backward()
                d_g_z1 = fake_output.mean()

                # Count all discriminator losses.
                d_loss = d_loss_real + d_loss_fake
                optimizer_D.step()

                ##############################################
                # (2) Update G network: min E(z)[log(1- D(z))]
                ##############################################
                # Set generator gradients to zero.
                global_model.zero_grad()

                fake_output = discriminator(fake, conditional)
                g_loss = adversarial_loss(fake_output, real_label)
                g_loss.backward()
                d_g_z2 = fake_output.mean()
                optimizer_G.step()

                # if batch_idx % 100 == 0:
                #     print(f"Epoch {epoch + 1}, G Loss: {g_loss.item()}, D Loss: {d_loss.item()}")
            else:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(images), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item()))
                batch_loss.append(loss.item())

        if args.model != 'cgan':
            loss_avg = sum(batch_loss)/len(batch_loss)
            print('\nTrain loss:', loss_avg)
            epoch_loss.append(loss_avg)
        # elif (epoch + 1) % 10 == 0:
        #     with torch.no_grad():
        #     # Switch model to eval mode.
        #         global_model.eval()  # Set the model to evaluation mode
        #         # Create a figure to display images
        #         fig, axs = plt.subplots(1, 10, figsize=(15, 3))
        #
        #         # Generate one image per label
        #         for label in range(10):
        #             z = torch.randn(1, 100, device=device)  # Generate random noise
        #             labels = torch.LongTensor([label]).to(device)  # Create a tensor for the label
        #
        #             with torch.no_grad():  # No need to track gradients
        #                 generated_image = global_model(z, labels).detach().cpu()
        #
        #             # Assuming output image is 1x28x28 (as in MNIST), adjust if different
        #             generated_image = generated_image.view(generated_image.size(1), generated_image.size(2), generated_image.size(3))  # Reshape image
        #             axs[label].imshow(generated_image.permute(1, 2, 0).squeeze(), cmap='gray' if generated_image.size(0) == 1 else None)
        #             axs[label].set_title(f'Label: {label}')
        #             axs[label].axis('off')
        #
        #         plt.tight_layout()
        #         plt.show()
        #        # vutils.save_image(sr.detach(), os.path.join("runs", f"GAN_epoch_{epoch}.png"), normalize=True)

    if args.model != 'cgan':
        torch.save(global_model.state_dict(), os.path.join("weights", f"GAN-last.pth"))
        # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

    # testing
    if args.model == 'cgan':
        save_cgan_generated_images(global_model, device, num_classes=args.num_classes, latent_dim=100)
    else:
        # testing
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print('Test on', len(test_dataset), 'samples')
        print("Test Accuracy: {:.2f}%".format(100 * test_acc))
