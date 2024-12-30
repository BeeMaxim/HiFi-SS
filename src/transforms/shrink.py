import torch
from torch import nn


class Shrink(nn.Module):
    """
    Shrink audio length
    """

    def __init__(self, factor):
        """
        Args:
            factor
        """
        super().__init__()

        self.factor = factor

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor if shape (1, T)
        Returns:
            x (Tensor): shrinked tensor.
        """
        x = x[:, : x.shape[1] // self.factor * self.factor]
        return x
