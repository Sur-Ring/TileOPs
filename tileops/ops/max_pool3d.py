"""Max Pooling 3D Forward Op.

Applies max pooling over an input signal composed of several input planes.

torch.compile support:
- MaxPooling3dFwdOp is registered via @torch.library.custom_op at module load time.
- Instances are looked up at runtime via _OP_REGISTRY keyed by id(instance).
- The instance key is a plain int so dynamo can trace through forward() without
  hitting unsupported Python side-effects.
"""

import functools
import weakref
from typing import List, Optional, Union

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pooling import MaxPooling3dFwdKernel
from tileops.ops.op import Op

__all__ = ["MaxPooling3dFwdOp"]

_OP_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


@functools.lru_cache(maxsize=64)
def _get_compiled(
    kernel_cls,
    batch: int,
    channels: int,
    in_d: int,
    in_h: int,
    in_w: int,
    kernel_size,
    stride,
    padding,
    dilation,
    dtype: torch.dtype,
    tune: bool,
):
    """Build and cache a compiled TileLang kernel. Called at most once per unique parameter set."""
    kern = kernel_cls(
        batch=batch,
        channels=channels,
        in_d=in_d,
        in_h=in_h,
        in_w=in_w,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        tune=tune,
    )
    return kern._compiled


def _register_max_pooling_3d_fwd_custom_op(op_cls) -> None:
    """Register MaxPooling3dFwdOp for torch.compile."""

    @torch.library.custom_op("top::pooling_max3d_fwd", mutates_args=())
    def _wrapped(x: torch.Tensor, _out_shape: List[int], instance_key: int) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(x)

    @_wrapped.register_fake
    def _(x: torch.Tensor, out_shape: List[int], _instance_key: int) -> torch.Tensor:
        return x.new_empty(out_shape)

    op_cls._wrapped = _wrapped


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

    _wrapped = None  # Set by _register_max_pooling_3d_fwd_custom_op at class definition

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
        # Register instance for torch.compile dispatch.
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self) -> dict[str, type[Kernel]]:
        return {"max_pooling_3d": MaxPooling3dFwdKernel}

    def _calculate_out_size(self, length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
        """Calculate output size for pooling."""
        return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def _unpack_params(self, in_d: int, in_h: int, in_w: int):
        """Return (out_d, out_h, out_w) given input spatial dims."""
        if isinstance(self.kernel_size, int):
            kd = kh = kw = self.kernel_size
        else:
            kd, kh, kw = self.kernel_size
        if isinstance(self.stride, int):
            sd = sh = sw = self.stride
        else:
            sd, sh, sw = self.stride
        if isinstance(self.padding, int):
            pd = ph = pw = self.padding
        else:
            pd, ph, pw = self.padding
        if isinstance(self.dilation, int):
            dd = dh = dw = self.dilation
        else:
            dd, dh, dw = self.dilation
        out_d = self._calculate_out_size(in_d, kd, sd, pd, dd)
        out_h = self._calculate_out_size(in_h, kh, sh, ph, dh)
        out_w = self._calculate_out_size(in_w, kw, sw, pw, dw)
        return out_d, out_h, out_w

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Direct kernel dispatch, bypassing validation. Called from custom_op and eager path."""
        batch, channels, in_d, in_h, in_w = x.shape
        compiled = _get_compiled(
            self.kernel_map["max_pooling_3d"],
            batch, channels, in_d, in_h, in_w,
            self.kernel_size, self.stride, self.padding, self.dilation,
            self.dtype, self.tune,
        )
        return compiled(x)

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

        if torch.compiler.is_compiling():
            batch, channels, in_d, in_h, in_w = x.shape
            out_d, out_h, out_w = self._unpack_params(in_d, in_h, in_w)
            return type(self)._wrapped(
                x, [batch, channels, out_d, out_h, out_w], self._instance_key
            )
        return self._eager_forward(x)

    def __repr__(self) -> str:
        return (
            f"MaxPooling3dFwdOp(kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, dtype={self.dtype})"
        )


_register_max_pooling_3d_fwd_custom_op(MaxPooling3dFwdOp)
