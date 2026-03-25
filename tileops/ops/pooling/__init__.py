"""Pooling ops package."""

from tileops.ops.pooling.avg_pool1d import AvgPooling1dOp
from tileops.ops.pooling.avg_pool2d import AvgPooling2dOp
from tileops.ops.pooling.avg_pool3d import AvgPooling3dOp
from tileops.ops.pooling.max_pool2d import MaxPooling2dFwdOp

__all__ = [
    "AvgPooling1dOp",
    "AvgPooling2dOp",
    "AvgPooling3dOp",
    "MaxPooling2dFwdOp",
]
