import torch
from torch.utils import data
from torchvision import datasets, transforms, utils
from pixelzoo.models.pixelcnn import PixelCNN
from pixelzoo.utils import EarlyStopping
import torch.optim as optim
import time, os, tqdm
import numpy as np


# Load train and test data
def load_data(batch_size=256, num_workers=4, shuffle=True):
    """
    Loads training and testing dataloaders
    """
    train_dataloader = data.DataLoader(datasets.MNIST(
        'data', train=True, download=True, transform=transforms.ToTensor()),
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       pin_memory=True)
    test_dataloader = data.DataLoader(datasets.MNIST(
        'data', train=False, download=True, transform=transforms.ToTensor()),
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      pin_memory=True)
    return train_dataloader, test_dataloader


train_dataloader, test_dataloader = load_data()
print("Dataloaders loaded!")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Initialize the model
model = PixelCNN(device=device)
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

    test_time = time.time() - test_time

    # sample images
    model.eval()
    with torch.no_grad():
        sampled_images = model.sample(64)
        utils.save_image(
            sampled_images,
            'images/pixelcnn/mnist/sample_{:02d}.png'.format(epoch + 1),
            nrow=12,
            padding=0)
    print(
        'epoch={}; nll_train={:.7f} bits/dim; nll_te={:.7f} bits/dim; time_train={:.1f}s; time_test={:.1f}s'
        .format(epoch + 1,
                np.mean(train_error) / np.log(2),
                np.mean(test_error) / np.log(2), train_time, test_time))
    epoch += 1
    if callback.early_stop(epoch + 1, np.mean(test_error) / np.log(2)):
        end_time = time.time()
        print(
            f'Early stopping after {epoch+1} epochs, training time: {(end_time-start_time)/60} minutes'
        )
        break
