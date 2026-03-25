"""Max Pooling 1D Forward Op.

Applies max pooling over an input signal composed of several input planes.

torch.compile support:
- MaxPooling1dFwdOp is registered via @torch.library.custom_op at module load time.
- Instances are looked up at runtime via _OP_REGISTRY keyed by id(instance).
- The instance key is a plain int so dynamo can trace through forward() without
  hitting unsupported Python side-effects.
"""

import functools
import weakref
from typing import List, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pooling import MaxPooling1dFwdKernel
from tileops.ops.op import Op

__all__ = ["MaxPooling1dFwdOp"]

_OP_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


@functools.lru_cache(maxsize=64)
def _get_compiled(
    kernel_cls,
    batch: int,
    channels: int,
    in_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: torch.dtype,
    tune: bool,
):
    """Build and cache a compiled TileLang kernel. Called at most once per unique parameter set."""
    kern = kernel_cls(
        batch=batch,
        channels=channels,
        in_length=in_length,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        tune=tune,
    )
    return kern._compiled


def _register_max_pooling_1d_fwd_custom_op(op_cls) -> None:
    """Register MaxPooling1dFwdOp for torch.compile."""

    @torch.library.custom_op("top::pooling_max1d_fwd", mutates_args=())
    def _wrapped(x: torch.Tensor, _out_shape: List[int], instance_key: int) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(x)

    @_wrapped.register_fake
    def _(x: torch.Tensor, out_shape: List[int], _instance_key: int) -> torch.Tensor:
        return x.new_empty(out_shape)

    op_cls._wrapped = _wrapped


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

    _wrapped = None  # Set by _register_max_pooling_1d_fwd_custom_op at class definition

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
        # Register instance for torch.compile dispatch.
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self) -> dict[str, type[Kernel]]:
        return {"max_pooling_1d": MaxPooling1dFwdKernel}

    def _calculate_out_size(self, length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
        """Calculate output size for pooling."""
        return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Direct kernel dispatch, bypassing validation. Called from custom_op and eager path."""
        batch, channels, in_length = x.shape
        compiled = _get_compiled(
            self.kernel_map["max_pooling_1d"],
            batch, channels, in_length,
            self.kernel_size, self.stride, self.padding, self.dilation,
            self.dtype, self.tune,
        )
        return compiled(x)

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

        if torch.compiler.is_compiling():
            batch, channels, in_length = x.shape
            out_length = self._calculate_out_size(
                in_length, self.kernel_size, self.stride, self.padding, self.dilation
            )
            return type(self)._wrapped(x, [batch, channels, out_length], self._instance_key)
        return self._eager_forward(x)

    def __repr__(self) -> str:
        return (
            f"MaxPooling1dFwdOp(kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, dtype={self.dtype})"
        )


_register_max_pooling_1d_fwd_custom_op(MaxPooling1dFwdOp)
