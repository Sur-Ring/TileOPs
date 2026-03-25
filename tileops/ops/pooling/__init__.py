"""Pooling ops package."""

from tileops.ops.pooling.max_pool1d import MaxPooling1dFwdOp
from tileops.ops.pooling.max_pool2d import MaxPooling2dFwdOp
from tileops.ops.pooling.max_pool3d import MaxPooling3dFwdOp

__all__ = [
    "MaxPooling1dFwdOp",
    "MaxPooling2dFwdOp",
    "MaxPooling3dFwdOp",
]
