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

from gan_evaluation import plot_cgan_generated_images, load_classifier, generate_images, classifier_accuracy, \
    calculate_emd, pca_images, plot_pca
from datetime import datetime

from sdv.metadata.single_table import SingleTableMetadata
from sdv_local.single_table.ctgan import CTGANSynthesizer, TVAESynthesizer
from utils import get_dataset, get_dataset_config, index_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from CGAN_PyTorch.cgan_pytorch.utils.common import configure, weights_init
from CGAN_PyTorch.cgan_pytorch.models.discriminator import get_discriminator


def save_model_with_timestamp(model, args, path="weights"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(),
               os.path.join('..', path, f"CGAN_{args.dataset}_{args.epochs}_{args.noise}_{args.lr}_{timestamp}.pth"))


def testing(args, global_model, test_dataset, device, train_dataset=None, save_model=False, plot=False):
    if args.model != 'ctgan' and args.model != 'tvae':
        testloader = DataLoader(test_dataset, batch_size=256, shuffle=True)
    # testing
    if args.model == 'cgan':
        # torch.save(global_model.state_dict(), os.path.join("weights", f"GAN-last.pth"))
        if plot:
            plot_cgan_generated_images(global_model, device, args.dataset, num_classes=args.num_classes,
                                       latent_dim=args.noise)
        real_preds = 0
        fake_preds = 0
        emd_real_images = torch.empty(0, ).to(device)
        emd_fake_images = torch.empty(0, ).to(device)
        labels_list = torch.empty(0, ).to(device)

        model = load_classifier(args.dataset)
        for batch_idx, (images, labels) in tqdm(enumerate(testloader), f"Testing: ", total=len(testloader)):
            images = images.to(device)
            labels = labels.to(device)
            fake_images = generate_images(global_model, device, labels, num_images=images.size(0),
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
    elif args.model != 'ctgan' and args.model != 'tvae':
        # testing
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print('Test on', len(test_dataset), 'samples')
        print("Test Accuracy: {:.2f}%".format(100 * test_acc))
    else:
        global_model.save(os.path.join("weights", f"{args.model}.pth"))
        # sample and save the dataframe to csv
        sample = global_model.sample(len(train_dataset))
        sample.to_csv(os.path.join("synthetic_datasets", f"sample_baseline_{args.model}.csv"), index=False)


def main(args):
    accuracies = []
    emds = []
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
        global_model = configure(args).to(device)
        if args.dataset == 'mnist':
            discriminator = get_discriminator('mnist', args.noise)
            discriminator.to(device)
            discriminator.train()

            # Set loss function
            adversarial_loss = torch.nn.MSELoss().to(device)

            # Set optimizers for both Generator and Discriminator
            optimizer_G = Adam(global_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
            optimizer_D = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        elif args.dataset == 'cifar':
            global_model.apply(weights_init)
            discriminator = get_discriminator('cifar', args.noise).to(device)
            discriminator.apply(weights_init)

            discriminator.train()
            adversarial_loss = BCELoss().to(device)

            optimizer_G = Adam(global_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
            optimizer_D = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        fixed_noise = torch.randn([batch_size, args.noise]).to(device)
        fixed_conditional = torch.randint(0, 1, (batch_size,)).to(device)
    elif args.model == 'ctgan' or args.model == 'tvae':
        if args.dataset == 'adult':
            config = get_dataset_config('adult_config.json')
            train_dataset = index_dataset(train_dataset, config)
            stm = SingleTableMetadata()
            has_id_column = config.pop("has_id")
            metadata = stm.load_from_dict(config)
            if args.model == 'tvae':
                global_model = TVAESynthesizer(metadata, train_dataset, cuda=True, epochs=300)
            else:
                global_model = CTGANSynthesizer(metadata, train_dataset, cuda=True, epochs=300)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    if args.model == 'ctgan' or args.model == 'tvae':
        global_model.fit(train_dataset)
    else:
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

            for batch_idx, (images, labels) in tqdm(enumerate(trainloader), f"Epoch {epoch + 1}: ", total=len(trainloader)):
                images = images.to(device)
                labels = labels.to(device)
                if args.model == 'cgan':
                    discriminator.train()
                    global_model.train()
                    btch_size = images.size(0)
                    real_label = torch.full((btch_size, 1), 1, dtype=images.dtype).to(device)
                    fake_label = torch.full((btch_size, 1), 0, dtype=images.dtype).to(device)

                    noise = torch.randn([btch_size, args.noise]).to(device) if args.dataset == 'mnist' else \
                        torch.randn(btch_size, args.noise, 1, 1, device=device)
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
                            epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                            100. * batch_idx / len(trainloader), loss.item()))
                    batch_loss.append(loss.item())

            if (epoch + 1) % args.testevery == 0:
                accuracy, emd = testing(args, global_model, test_dataset, device, train_dataset)
                accuracies.append(accuracy)
                emds.append(emd)
            if args.model != 'cgan':
                loss_avg = sum(batch_loss) / len(batch_loss)
                print('\nTrain loss:', loss_avg)
                epoch_loss.append(loss_avg)
            # elif (epoch + 1) % 1 == 0:
            #     with torch.no_grad():
            #         # # Switch model to eval mode.
            #         # global_model.eval()  # Set the model to evaluation mode
            #         # # Create a figure to display images
            #         # fig, axs = plt.subplots(1, 10, figsize=(15, 3))
            #         #
            #         # # Generate one image per label
            #         # for label in range(10):
            #         #     z = torch.randn(1, args.noise, device=device)  # Generate random noise
            #         #     labels = torch.LongTensor([label]).to(device)  # Create a tensor for the label
            #         #
            #         #     with torch.no_grad():  # No need to track gradients
            #         #         generated_image = global_model(z, labels).detach().cpu()
            #         #
            #         #     # Assuming output image is 1x28x28 (as in MNIST), adjust if different
            #         #     generated_image = generated_image.view(generated_image.size(1), generated_image.size(2),
            #         #                                            generated_image.size(3))  # Reshape image
            #         #     axs[label].imshow(generated_image.permute(1, 2, 0).squeeze(),
            #         #                       cmap='gray' if generated_image.size(0) == 1 else None)
            #         #     axs[label].set_title(f'Label: {label}')
            #         #     axs[label].axis('off')
            #         #
            #         # plt.tight_layout()
            #         # plt.show()
            #         save_cgan_generated_images(global_model, device, args.dataset, num_classes=args.num_classes,
            #                                    latent_dim=args.noise)
    if args.model != 'cgan' and args.model != 'ctgan' and args.model != 'tvae':
        # Plot loss
        plt.figure()
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.xlabel('epochs')
        plt.ylabel('Train loss')
        plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                     args.epochs))
    acc, emd = testing(args, global_model, test_dataset, device, train_dataset, plot=True)
    accuracies.append(acc)
    emds.append(emd)

    return accuracies, emds


if __name__ == '__main__':
    args = args_parser()
    main(args)
