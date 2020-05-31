import torch
from torch.utils import data
from torchvision import datasets, transforms, utils
from pixelzoo.models import PixelCNN, GatedPixelCNN
from pixelzoo.utils import EarlyStopping
import torch.optim as optim
import time, os
import numpy as np
import argparse


# Load train and test data
def load_data(dataset='cifar10', batch_size=512, num_workers=4, shuffle=True):
    """
    Loads training and testing dataloaders
    """
    if dataset == 'mnist':
        train_dataloader = data.DataLoader(
            datasets.MNIST('data',
                           train=True,
                           download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True)
        test_dataloader = data.DataLoader(datasets.MNIST(
            'data',
            train=False,
            download=True,
            transform=transforms.ToTensor()),
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          pin_memory=True)

    else:
        train_dataloader = data.DataLoader(
            datasets.CIFAR10('data',
                             train=True,
                             download=True,
                             transform=transforms.ToTensor()),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True)
        test_dataloader = data.DataLoader(datasets.CIFAR10(
            'data',
            train=False,
            download=True,
            transform=transforms.ToTensor()),
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          pin_memory=True)

    return train_dataloader, test_dataloader


train_dataloader, test_dataloader = load_data()
print("Dataloaders loaded!")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    # Initialize the model
    if args.model == 'pixelcnn':
        model = PixelCNN(logits_dist=args.logits_dist, device=device)
    elif args.model == 'gatedpixelcnn':
        model = GatedPixelCNN(device=device)

    model.to(device=device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    # Initialze the EarlyStopping Callback
    callback = EarlyStopping()
    start_time = time.time()
    # Start the training loop
    epoch = 1
    print("Start training!")

    while True:

        train_error = []
        train_time = time.time()
        model.train()
        for step, (images, labels) in enumerate(train_dataloader):
            # nll of the batched data
            loss = model.nll(images.to(device))
            train_error.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del loss
        train_time = time.time() - train_time

        # compute error on test set
        test_error = []
        test_time = time.time()
        model.eval()
        with torch.no_grad():

            for images, labels in test_dataloader:
                # nll of the test data batch
                loss = model.nll(images.to(device))
                test_error.append(loss.item())

        del loss
        test_time = time.time() - test_time

        sample_start = time.time()
        # ========== sample images =========== #
        model.eval()
        with torch.no_grad():
            sampled_images = model.sample(64)
            utils.save_image(
                sampled_images,
                'images/gatedpixelcnn/cifar10/0.0001__sample_{:02d}.png'.
                format(epoch),
                nrow=8,
                padding=0)
        del sampled_images

        sample_time = time.time() - sample_start

        print(
            'epoch={}; nll_train={:.7f} bits/dim; nll_te={:.7f} bits/dim; time_train={:.1f}s; time_test={:.1f}s: sampling time={:.1f}'
            .format(epoch,
                    np.mean(train_error) / np.log(2),
                    np.mean(test_error) / np.log(2), train_time, test_time,
                    sample_time))
        epoch += 1
        if callback.early_stop(epoch, np.mean(test_error) / np.log(2)):
            total_time = time.time() - start_time
            print('Early stopping after {} epochs, training time: {} minutes'.
                  format(epoch, total_time / 60))
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='pixelcnn',
        help=
        'The model you wish to train: pixelcnn, pixelcnn++, gatedpixelcnn, conditionalpixelcnn, pixelsnail'
    )
    parser.add_argument(
        '--logits_dist',
        type=str,
        default='categorical',
        help='Distribution over output pixels: categorical or sigmoid')
    parser.add_argument('--dataset',
                        type=str,
                        default='mnist',
                        help='Dataset to train on: mnist or cifar')

    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    load_data(args.dataset, args.batch_size)

    main(args)
