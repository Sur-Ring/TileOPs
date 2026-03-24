"""Max Pooling Forward Op.

Applies max pooling over an input signal composed of several input planes.
"""

from typing import Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pooling import MaxPoolingFwdKernel

from .op import Op

__all__ = ["MaxPoolingFwdOp"]


# MaxPool2dOp
class MaxPoolingFwdOp(Op):
    """Max Pooling Forward Operator.

    Applies max pooling over an input signal composed of several input planes.

    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Padding added to input. If None, defaults to 0.
        dilation: Dilation of the pooling window. If None, defaults to 1.
        dtype: Data type (float16 or bfloat16).
        kernel_map: Optional dict mapping kernel names to Kernel classes.
        tune: Whether to run autotuning.

    Example:
        >>> op = MaxPoolingFwdOp(kernel_size=2, stride=2)
        >>> x = torch.randn(1, 512, 4096, dtype=torch.float16, device="cuda")
        >>> y = op(x)
        >>> y.shape
        torch.Size([1, 256, 4096])
    """

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: Optional[int] = None,
        dilation: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[dict[str, type[Kernel]]] = None,
        tune: bool = False,
    ):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding if padding is not None else 0
        self.dilation = dilation if dilation is not None else 1
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> dict[str, type[Kernel]]:
        return {"max_pooling": MaxPoolingFwdKernel}

    def _calculate_out_size(self, length: int) -> int:
        """Calculate output size for pooling."""
        return (
            length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of max pooling.

        Args:
            x: Input tensor of shape (batch, in_seq_len, dim).

        Returns:
            Output tensor of shape (batch, out_seq_len, dim).
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, dim), got {x.ndim}D")

        batch, in_seq_len, dim = x.shape
        out_seq_len = self._calculate_out_size(in_seq_len)

        # Validate padding doesn't cause negative output
        if out_seq_len <= 0:
            raise ValueError(
                f"Output sequence length is {out_seq_len}, which is invalid. "
                f"Check kernel_size, stride, padding, and dilation values."
            )

        kernel = self.kernel_map["max_pooling"](
            batch=batch,
            in_seq_len=in_seq_len,
            dim=dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            dtype=self.dtype,
            tune=self.tune,
        )

        return kernel.forward(x)

    def __repr__(self) -> str:
        return (
            f"MaxPoolingFwdOp(kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, dtype={self.dtype})"
        )
