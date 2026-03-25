"""Average Pooling 1D Forward Op.

Applies average pooling over an input signal composed of several input planes.
"""

from typing import Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pooling import AvgPooling1dKernel
from tileops.ops.op import Op

__all__ = ["AvgPooling1dOp"]


class AvgPooling1dOp(Op):
    """Average Pooling 1D Forward Operator.

    Applies average pooling over an input signal composed of several input planes.

    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Padding added to input. If None, defaults to 0.
        dilation: Dilation of the pooling window. If None, defaults to 1.
        dtype: Data type (float16 or bfloat16).
        accum_dtype: Accumulator dtype for sum reduction (float32 recommended).
        kernel_map: Optional dict mapping kernel names to Kernel classes.
        tune: Whether to run autotuning.

    Example:
        >>> op = AvgPooling1dOp(kernel_size=2, stride=2)
        >>> x = torch.randn(1, 3, 100, dtype=torch.float16, device="cuda")
        >>> y = op(x)
        >>> y.shape
        torch.Size([1, 3, 50])
    """

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: Optional[int] = None,
        dilation: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        kernel_map: Optional[dict[str, type[Kernel]]] = None,
        tune: bool = False,
    ):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding if padding is not None else 0
        self.dilation = dilation if dilation is not None else 1
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> dict[str, type[Kernel]]:
        return {"avg_pooling_1d": AvgPooling1dKernel}

    def _calculate_out_size(self, length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
        """Calculate output size for pooling."""
        return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of average pooling 1D.

        Args:
            x: Input tensor of shape (batch, channels, in_seq_len).

        Returns:
            Output tensor of shape (batch, channels, out_seq_len).
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, channels, in_seq_len), got {x.ndim}D")

        batch, channels, in_seq_len = x.shape

        out_seq_len = self._calculate_out_size(
            in_seq_len, self.kernel_size, self.stride, self.padding, self.dilation
        )

        # Validate padding doesn't cause negative output
        if out_seq_len <= 0:
            raise ValueError(
                f"Output sequence length is {out_seq_len}, which is invalid. "
                f"Check kernel_size, stride, padding, and dilation values."
            )

        kernel = self.kernel_map["avg_pooling_1d"](
            batch=batch,
            channels=channels,
            in_seq_len=in_seq_len,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            dtype=self.dtype,
            accum_dtype=self.accum_dtype,
            tune=self.tune,
        )

        return kernel.forward(x)

    def __repr__(self) -> str:
        return (
            f"AvgPooling1dOp(kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, dtype={self.dtype})"
        )
