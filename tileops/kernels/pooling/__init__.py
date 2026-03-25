"""Pooling kernels package."""

from tileops.kernels.pooling.max_pool1d import MaxPooling1dFwdKernel
from tileops.kernels.pooling.max_pool2d import MaxPooling2dFwdKernel
from tileops.kernels.pooling.max_pool3d import MaxPooling3dFwdKernel

__all__ = [
    "MaxPooling1dFwdKernel",
    "MaxPooling2dFwdKernel",
    "MaxPooling3dFwdKernel",
]
