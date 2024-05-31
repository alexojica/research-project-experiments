#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.idxs = idxs
        if args.dataset == 'adult':
            self.dataset = dataset
        else:
            self.trainloader, self.validloader, self.testloader = self.train_val_test(
                dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        if args.model == 'cgan':
            # Set loss function
            self.adversarial_loss = torch.nn.MSELoss().to(self.device)

            self.fixed_noise = torch.randn([args.local_bs, 100]).to(self.device)
            self.fixed_conditional = torch.randint(0, 1, (args.local_bs,)).to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        device = self.device
        if self.args.model == 'cgan':
            generator = model['generator']
            discriminator = model['discriminator']
            generator.train()
            discriminator.train()
            adversarial_loss = self.adversarial_loss

            optimizer_G = Adam(model['generator'].parameters(), lr=self.args.lr, betas=(0.5, 0.999))
            optimizer_D = Adam(model['discriminator'].parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        elif self.args.model != 'ctgan' and self.args.model != 'tvae':
            model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.model != 'cgan' and self.args.model != 'ctgan' and self.args.model != 'tvae':
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                            momentum=0.5)
            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                             weight_decay=1e-4)

        if self.args.model == 'ctgan' or self.args.model == 'tvae':
            model.fit(self.dataset.iloc[list(self.idxs)])
            return model.get_weights(), 0, model._data_processor
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                if self.args.model == 'cgan':
                    discriminator.train()
                    generator.train()
                    btch_size = images.size(0)
                    real_label = torch.full((btch_size, 1), 1, dtype=images.dtype).to(device)
                    fake_label = torch.full((btch_size, 1), 0, dtype=images.dtype).to(device)

                    noise = torch.randn([btch_size, self.args.noise]).to(device)
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
                    fake = generator(noise, conditional)
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
                    generator.zero_grad()

                    fake_output = discriminator(fake, conditional)
                    g_loss = adversarial_loss(fake_output, real_label)
                    g_loss.backward()
                    d_g_z2 = fake_output.mean()
                    optimizer_G.step()
                    batch_loss.append((g_loss.item() + d_loss.item()) / 2)
                else:
                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        if self.args.model == 'cgan':
            return {'generator': generator.state_dict(), 'discriminator': discriminator.state_dict()}, sum(epoch_loss) / len(epoch_loss)
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, args):
        """ Returns the inference accuracy and loss.
        """

        if args.model == 'cgan':
            model['generator'].eval()
            model['discriminator'].eval()
        else:
            model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            # fig, axs = plt.subplots(1, 10, figsize=(15, 3))

            if args.model == 'cgan':
                # generate an image for each label with the generator
                for i in range(10):
                    # z = torch.randn(1, 100, device=self.device)  # Generate random noise
                    # labels = torch.LongTensor([i]).to(self.device)  # Create a tensor for the label
                    # with torch.no_grad():
                    #     generated_image = model['generator'](z, labels).detach().cpu()
                    # generated_image = generated_image.view(generated_image.size(1), generated_image.size(2),
                    #                                        generated_image.size(3))  # Reshape image
                    # axs[i].imshow(generated_image.permute(1, 2, 0).squeeze(),
                    #                   cmap='gray' if generated_image.size(0) == 1 else None)
                    # axs[i].set_title(f'Label: {i}')
                    # axs[i].axis('off')
                    total = 1
            else:
                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        # plt.tight_layout()
        # plt.show()
        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
