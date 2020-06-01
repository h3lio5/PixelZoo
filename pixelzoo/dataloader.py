from torch.utils import data
from torchvision import datasets, transforms


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