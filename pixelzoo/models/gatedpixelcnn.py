import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv2D(nn.Module):
    """
    Convolution network architecture as described in the paper ().
    The 'blind spot' is removed by combining the vertical and horizontal convolution network stacks.
    Part of code taken from https://github.com/pbloem/pixel-models/blob/master/layers.py 
    """
    def __init__(self,
                 mask_type,
                 in_channels,
                 out_channels,
                 colors=3,
                 self_connection=True,
                 gates=True,
                 hv_connection=True,
                 residual_connection=True,
                 k=7,
                 padding=3):
        """
        """
        super().__init__()
        self.hv_connection = hv_connection
        self.gates = gates
        self.residual_connection = residual_connection
        self.vStack = nn.Conv2d(in_channels,
                                out_channels * 2,
                                kernel_size=k,
                                padding=padding,
                                bias=False)
        self.hStack = nn.Conv2d(in_channels,
                                out_channels * 2,
                                kernel_size=(1, k),
                                padding=(0, padding),
                                bias=False)
        # Connects the output of the vertical stack to the input of the horizontal stack. If we had connected
        # the output of the horizontal stack into the vertical stack, it would be able to use information
        # about pixels that are below or to the right of the current pixel which would break the conditional distribution.
        self.vStack_to_hStack = nn.Conv2d(out_channels * 2,
                                          out_channels * 2,
                                          kernel_size=1,
                                          bias=False,
                                          padding=0,
                                          groups=colors)
        self.resNet = nn.Conv2d(out_channels,
                                out_channels,
                                kernel_size=1,
                                padding=0,
                                bias=False,
                                groups=colors)

        self.register_buffer('vmask', torch.ones_like(self.vStack.weight))
        self.register_buffer('hmask', torch.ones_like(self.hStack.weight))

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 + 1:, :] = 0
        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2 + 1:] = 0
        if mask_type == 'A':
            self.hmask[:, :, :, k // 2] = 0

        # Add connections to "previous" colors (G is allowed to see R, and B is allowed to see R and G)
        m = k // 2  # index of the middle of the convolution
        channels_per_color = out_channels // colors  # channels per color

        #
        for c in range(0, colors):
            f, t = c * channels_per_color, (c + 1) * channels_per_color

            if f > 0:
                self.hmask[f:t, :f, 0, m] = 1

            # Connections to "current" colors (but not "future colors", R is not allowed to see G and B)
            if self_connection:
                self.hmask[f:t, :f + channels_per_color, 0, m] = 1

    def forward(self, x):
        """
        """

        v_input, h_input = x

        # Masking out the kernel weights
        self.vStack.weight.data *= self.vmask
        self.hStack.weight.data *= self.hmask

        v_output = self.vStack(v_input)
        h_output = self.hStack(h_input)

        # Combining the output of vertical stack with horizontal stack to allow the flow of information about pixels above
        # while predicting the current pixel.
        if self.hv_connection:
            h_output = h_output + self.vStack_to_hStack(v_output)

        # Replace ReLU activation function with multiplicative gating to enable the network to model
        # more complex interactions.
        if self.gates:
            v_output = self.gate(v_output)
            h_output = self.gate(h_output)

        # Add a residual connection between the input and outputs of the horizontal stack.
        # Residual connection is present in vertical stack since no notable improvement was observed as given in the paper.
        if self.residual_connection:
            h_output = h_input + self.resNet(h_output)

        return (v_output, h_output)

    def gate(self, x):
        """
        Takes a batch x channels x rest... tensor and applies an LTSM-style gated activation.
        - The top half of the channels are fed through a tanh activation, functioning as the activated neurons
        - The bottom half are fed through a sigmoid, functioning as a gate
        - The two are element-wise multiplied, and the result is returned.
        """

        top, bottom = torch.chunk(x, 2, dim=1)

        return torch.tanh(top) * torch.sigmoid(bottom)


class GatedPixelCNN(nn.Module):
    """
    """
    def __init__(self, c=3, channels=63, n_layers=7, device='cpu'):
        super().__init__()

        self.conv1 = nn.Conv2d(c, channels, 1, groups=c)
        model = [
            GatedConv2D('A',
                        channels,
                        channels,
                        self_connection=False,
                        residual_connection=False)
        ]
        for _ in range(n_layers - 1):
            model.extend([GatedConv2D('B', channels, channels)])
        self.model = nn.Sequential(*model)
        self.final_layer = nn.Conv2d(channels, 256 * c, 1, groups=c)

        self.device = device

    def net(self, x):
        """
        """
        b, c, h, w = x.size()

        x = self.conv1(x)
        h_output, v_output = self.model((x, x))
        target = self.final_layer(h_output)

        return target.view(b, c, 256, h, w).transpose(1, 2)

    def nll(self, input):
        """
        """

        target = self.net(input)
        return F.cross_entropy(target, (input.data * 255).long())

    def sample(self, n):
        """
        
        """

        with torch.no_grad():
            for h in range(32):
                for w in range(32):
                    for c in range(3):
                        logits = self.net(samples)[:, :, c, h, w]
                        probs = F.softmax(logits, dim=1)
                        samples[:, c, h,
                                w] = torch.multinomial(probs, 1).float() / 255.

        return samples.cpu()
