"""
This file contains various layers used to implement 
pixel-based autoregressive generative models
"""
import torch.nn as nn
import torch


class MaskedConv2D(nn.Conv2d):
    """
    Vanilla masked conv layer which creates a blind spot in the receptive field when stacked. 
    """
    def __init__(self, mask_type):

        assert mask_type == 'A' or mask_type == 'B'
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if mask_type == 'B':
            self.mask[:, :, k // 2, k // 2] = 1
