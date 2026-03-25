"""Pooling kernels package."""

from tileops.kernels.pooling.avg_pool1d import AvgPooling1dKernel
from tileops.kernels.pooling.avg_pool2d import AvgPooling2dKernel
from tileops.kernels.pooling.avg_pool3d import AvgPooling3dKernel
from tileops.kernels.pooling.max_pool1d import MaxPooling1dFwdKernel
from tileops.kernels.pooling.max_pool2d import MaxPooling2dFwdKernel
from tileops.kernels.pooling.max_pool3d import MaxPooling3dFwdKernel

__all__ = [
    "AvgPooling1dKernel",
    "AvgPooling2dKernel",
    "AvgPooling3dKernel",
    "MaxPooling1dFwdKernel",
    "MaxPooling2dFwdKernel",
    "MaxPooling3dFwdKernel",
]
