"""
This module contains the model architecture proposed in the paper PixelRNN (Oord e.t. al.)
"""

import torch
import torch.nn as nn


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