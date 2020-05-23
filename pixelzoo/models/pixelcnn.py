"""
This module contains the model architecture proposed in the paper PixelRNN (Oord e.t. al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2D(nn.Conv2d):
    """
    Vanilla masked conv layer which creates a blind spot in the receptive field when stacked. 
    """
    def __init__(self, mask_type, *args, **kwargs):
        """
        Arguments:
            mask_type(str): The type of mask to be applied.
                            mask 'A': Mask out the "current pixel" (ie. the middle of the convolution). In the first 
                                      layer, this should be masked out, since it connects to the value we're trying to predict. 
                            mask 'B': In higher layers, it should not be masked as it conveys the intermediate representations 
                                      we're building up.
            args(tuple): passed to the conv2d layer
            kwargs(dict): passed to the conv2d layer
        """
        assert mask_type == 'A' or mask_type == 'B'
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def forward(self, input):
        """
        Arguments:
            input(torch.Tensor): Batch of images.
                                 shape: (batch_size,C,H,W)
        """
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def create_mask(self, mask_type):
        """
        Creates a mask of the required type.
        Arguments:
            mask_type(str): type of the mask, i.e. 'A' or 'B'.
        """
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if mask_type == 'B':
            self.mask[:, :, k // 2, k // 2] = 1


class PixelCNN(nn.Module):
    """
    Simple PixelCNN model architecture
    """
    def __init__(self):

        super().__init__()
        model = [
            MaskedConv2D('A', 1, 64, 7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        ]
        for _ in range(8):
            model.extend([
                MaskedConv2D('B', 64, 64, 7, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ])
        model.append(MaskedConv2D('B', 64, 1, 7, padding=3, bias=False))
        self.net = nn.Sequential(*model)

    def nll(self, input):
        """
        Arguments:
            input(torch.Tensor): batch of images.
                                 shape: (batch_size,C,H,W)
        Returns:
            Negative log likelihood of the data
        """
        logits = self.net(input)
        return F.binary_cross_entropy_with_logits(logits, input)

    def sample(self, n):
        """
        Arguments:
            n(int): Number of images to sample
        Returns:
            samples(torch.Tensor): a set of sampled images.
                                   shape: (n,c,h,w)
        """

        samples = torch.zeros(n, 1, 28, 28).to(self.device)
        with torch.no_grad():
            for r in range(28):
                for c in range(28):
                    logits = self.net(samples)[:, :, r, c]
                    probs = torch.sigmoid(logits)
                    samples[:, :, r, c] = torch.bernoulli(probs)

        return samples.cpu()
