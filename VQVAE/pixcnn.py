#代码来自 https://zhuanlan.zhihu.com/p/461693342
import torch
from torch import nn


class MaskedConv2d(nn.Conv2d):
    """
    Implements a conv2d with mask applied on its weights.

    Args:
        mask_type (str): the mask type, 'A' or 'B'.
        in_channels (int) – Number of channels in the input image.
        out_channels (int) – Number of channels produced by the convolution.
        kernel_size (int or tuple) – Size of the convolving kernel
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.mask_type = mask_type

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        mask = torch.zeros(kernel_size)
        mask[:kernel_size[0] // 2, :] = 1.0
        mask[kernel_size[0] // 2, :kernel_size[1] // 2] = 1.0
        if self.mask_type == "B":
            mask[kernel_size[0] // 2, kernel_size[1] // 2] = 1.0
        self.register_buffer('mask', mask[None, None])

    def forward(self, x):
        self.weight.data *= self.mask  # mask weights
        return super().forward(x)

class ResidualBlock(nn.Module):
    """
    Residual Block: conv1x1 -> conv3x3 -> conv1x1
    """

    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True)
        )
        # masked conv2d
        self.conv2 = nn.Sequential(
            MaskedConv2d("B", in_channels // 2, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return inputs + x

class PixelCNN(nn.Module):
    """
    PixelCNN model
    """

    def __init__(self, in_channels=1, channels=128, out_channels=1, n_residual_blocks=5):
        super().__init__()

        # we use maskedconv "A" for the first layer
        self.stem = nn.Sequential(
            MaskedConv2d("A", in_channels, channels, 7, padding=3),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(n_residual_blocks)]
        )
        # 这里我采用了两个3x3 conv，论文采用的是1x1 conv
        self.head = nn.Sequential(
            MaskedConv2d("B", channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d("B", channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.head(x)
        return x

class PixelCNNWithEmbedding(PixelCNN):
    def __init__(self, n_embedding, embedding_dim, channels=128, out_channels=1, n_residual_blocks=5):
        super().__init__(in_channels=embedding_dim, channels=channels, out_channels=out_channels, n_residual_blocks=n_residual_blocks)
        self.embedding = nn.Embedding(n_embedding, embedding_dim)
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = super().forward(x)
        # Assuming x is of shape [batch_size, channels, height, width]
        # We can apply the embedding to the output
        # Here we assume that the output channels match the embedding dimension
        return x
