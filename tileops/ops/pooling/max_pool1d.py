"""Max Pooling 1D Forward Op.

Applies max pooling over an input signal composed of several input planes.
"""

from typing import Optional, Union

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pooling import MaxPooling1dFwdKernel

from tileops.ops.op import Op

__all__ = ["MaxPooling1dFwdOp"]


class MaxPooling1dFwdOp(Op):
    """Max Pooling 1D Forward Operator.

    Applies max pooling over an input signal composed of several input planes.

    Args:
        kernel_size: Size of the pooling window (int).
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Padding added to input. If None, defaults to 0.
        dilation: Dilation of the pooling window. If None, defaults to 1.
        dtype: Data type (float16 or bfloat16).
        kernel_map: Optional dict mapping kernel names to Kernel classes.
        tune: Whether to run autotuning.

    Example:
        >>> op = MaxPooling1dFwdOp(kernel_size=2, stride=2)
        >>> x = torch.randn(1, 3, 224, dtype=torch.float16, device="cuda")
        >>> y = op(x)
        >>> y.shape
        torch.Size([1, 3, 112])
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
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        # Maps input shape → compiled TileLang kernel for zero-overhead hot path.
        self._compiled_cache: dict = {}

    @property
    def default_kernel_map(self) -> dict[str, type[Kernel]]:
        return {"max_pooling_1d": MaxPooling1dFwdKernel}

    def _calculate_out_size(self, length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
        """Calculate output size for pooling."""
        return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of max pooling 1D.

        Args:
            x: Input tensor of shape (batch, channels, length).

        Returns:
            Output tensor of shape (batch, channels, out_length).
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, channels, length), got {x.ndim}D")

        # Hot path: compiled kernel is cached per input shape to minimise Python overhead.
        shape = x.shape
        compiled = self._compiled_cache.get(shape)
        if compiled is None:
            compiled = self._build_compiled(shape)
        return compiled(x)

    def _build_compiled(self, shape: torch.Size):
        """Build and cache the compiled kernel for the given input shape."""
        batch, channels, in_length = shape

        out_length = self._calculate_out_size(
            in_length, self.kernel_size, self.stride, self.padding, self.dilation
        )

        if out_length <= 0:
            raise ValueError(
                f"Output length is {out_length}, which is invalid. "
                f"Check kernel_size, stride, padding, and dilation values."
            )

        kern = self.kernel_map["max_pooling_1d"](
            batch=batch,
            channels=channels,
            in_length=in_length,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            dtype=self.dtype,
            tune=self.tune,
        )
        compiled = kern._compiled
        self._compiled_cache[shape] = compiled
        return compiled

    def __repr__(self) -> str:
        return (
            f"MaxPooling1dFwdOp(kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, dtype={self.dtype})"
        )
