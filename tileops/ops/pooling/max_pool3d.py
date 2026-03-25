"""Max Pooling 3D Forward Op.

Applies max pooling over an input signal composed of several input planes.
"""

from typing import Optional, Union

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pooling import MaxPooling3dFwdKernel

from tileops.ops.op import Op

__all__ = ["MaxPooling3dFwdOp"]


class MaxPooling3dFwdOp(Op):
    """Max Pooling 3D Forward Operator.

    Applies max pooling over an input signal composed of several input planes.

    Args:
        kernel_size: Size of the pooling window. Can be an int for cubic kernel
            or (kernel_d, kernel_h, kernel_w) for non-cubic kernel.
        stride: Stride of the pooling window. If None, defaults to kernel_size.
            Can be an int or (stride_d, stride_h, stride_w).
        padding: Padding added to input. If None, defaults to 0.
            Can be an int or (pad_d, pad_h, pad_w).
        dilation: Dilation of the pooling window. If None, defaults to 1.
            Can be an int or (dilation_d, dilation_h, dilation_w).
        dtype: Data type (float16 or bfloat16).
        kernel_map: Optional dict mapping kernel names to Kernel classes.
        tune: Whether to run autotuning.

    Example:
        >>> op = MaxPooling3dFwdOp(kernel_size=2, stride=2)
        >>> x = torch.randn(1, 3, 16, 16, 16, dtype=torch.float16, device="cuda")
        >>> y = op(x)
        >>> y.shape
        torch.Size([1, 3, 8, 8, 8])
    """

    def __init__(
        self,
        kernel_size: Union[int, tuple[int, int, int]],
        stride: Optional[Union[int, tuple[int, int, int]]] = None,
        padding: Optional[Union[int, tuple[int, int, int]]] = None,
        dilation: Optional[Union[int, tuple[int, int, int]]] = None,
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
        return {"max_pooling_3d": MaxPooling3dFwdKernel}

    def _calculate_out_size(self, length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
        """Calculate output size for pooling."""
        return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of max pooling 3D.

        Args:
            x: Input tensor of shape (batch, channels, depth, height, width).

        Returns:
            Output tensor of shape (batch, channels, out_d, out_h, out_w).
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (batch, channels, depth, height, width), got {x.ndim}D")

        # Hot path: compiled kernel is cached per input shape to minimise Python overhead.
        shape = x.shape
        compiled = self._compiled_cache.get(shape)
        if compiled is None:
            compiled = self._build_compiled(shape)
        return compiled(x)

    def _build_compiled(self, shape: torch.Size):
        """Build and cache the compiled kernel for the given input shape."""
        batch, channels, in_d, in_h, in_w = shape

        # Handle kernel_size as int or tuple
        if isinstance(self.kernel_size, int):
            kernel_d = kernel_h = kernel_w = self.kernel_size
        else:
            kernel_d, kernel_h, kernel_w = self.kernel_size

        # Handle stride
        if isinstance(self.stride, int):
            stride_d = stride_h = stride_w = self.stride
        else:
            stride_d, stride_h, stride_w = self.stride

        # Handle padding
        if isinstance(self.padding, int):
            padding_d = padding_h = padding_w = self.padding
        else:
            padding_d, padding_h, padding_w = self.padding

        # Handle dilation
        if isinstance(self.dilation, int):
            dilation_d = dilation_h = dilation_w = self.dilation
        else:
            dilation_d, dilation_h, dilation_w = self.dilation

        out_d = self._calculate_out_size(in_d, kernel_d, stride_d, padding_d, dilation_d)
        out_h = self._calculate_out_size(in_h, kernel_h, stride_h, padding_h, dilation_h)
        out_w = self._calculate_out_size(in_w, kernel_w, stride_w, padding_w, dilation_w)

        if out_d <= 0:
            raise ValueError(
                f"Output depth is {out_d}, which is invalid. "
                f"Check kernel_size, stride, padding, and dilation values."
            )
        if out_h <= 0:
            raise ValueError(
                f"Output height is {out_h}, which is invalid. "
                f"Check kernel_size, stride, padding, and dilation values."
            )
        if out_w <= 0:
            raise ValueError(
                f"Output width is {out_w}, which is invalid. "
                f"Check kernel_size, stride, padding, and dilation values."
            )

        kern = self.kernel_map["max_pooling_3d"](
            batch=batch,
            channels=channels,
            in_d=in_d,
            in_h=in_h,
            in_w=in_w,
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
            f"MaxPooling3dFwdOp(kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, dtype={self.dtype})"
        )
