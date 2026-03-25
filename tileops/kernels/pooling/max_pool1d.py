"""Max Pooling 1D Forward Kernel using TileLang.

Applies max pooling over an input signal composed of several input planes.
The input is 3D tensor (batch, channels, length).
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["MaxPooling1dFwdKernel"]


def _calculate_out_size(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    """Calculate output size for pooling."""
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@functools.lru_cache(maxsize=32)
def _max_pooling_1d_fwd_kernel(
    batch: int,
    channels: int,
    in_length: int,
    out_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: str,
) -> Callable:
    # Flat total number of output elements processed by the kernel.
    total = batch * channels * out_length

    @tilelang.jit(out_idx=[1])
    def _func(threads: int) -> None:
        @T.prim_func
        def main(
            x: T.Tensor((batch, channels, in_length), dtype),
            y: T.Tensor((batch, channels, out_length), dtype),
        ) -> None:
            # Flat 1-D grid: each block processes `threads` output elements.
            # Single-pass: write directly to y[b, c, ol] inside the Serial loop.
            with T.Kernel(T.ceildiv(total, threads), threads=threads) as pid:
                for i in T.Parallel(threads):
                    lin = pid * threads + i
                    if lin < total:
                        ol = lin % out_length
                        c  = (lin // out_length) % channels
                        b  = lin // (out_length * channels)
                        y[b, c, ol] = T.cast(-float("inf"), dtype)
                        for k in T.Serial(kernel_size):
                            l_idx = ol * stride - padding + k * dilation
                            if 0 <= l_idx < in_length:
                                y[b, c, ol] = T.max(y[b, c, ol], x[b, c, l_idx])

        return main

    return _func


class MaxPooling1dFwdKernel(Kernel):
    """Max Pooling 1D Forward Kernel.

    Supports SM80+ architectures. Applies max pooling with configurable
    kernel_size, stride, padding, and dilation.

    Args:
        batch: Batch size.
        channels: Number of channels.
        in_length: Input length.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Padding added to input. If None, defaults to 0.
        dilation: Dilation of the pooling window. If None, defaults to 1.
        dtype: Data type (float16 or bfloat16).
        config: Optional kernel configuration.
        tune: Whether to run autotuning.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        channels: int,
        in_length: int,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: Optional[int] = None,
        dilation: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.channels = channels
        self.in_length = in_length
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding if padding is not None else 0
        self.dilation = dilation if dilation is not None else 1
        self.dtype = dtype

        # Calculate output length
        self.out_length = _calculate_out_size(
            self.in_length, self.kernel_size, self.stride, self.padding, self.dilation
        )

        self.kernel = _max_pooling_1d_fwd_kernel(
            self.batch,
            self.channels,
            self.in_length,
            self.out_length,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.dtype_str,
        )
        self.init_config(config, tune)
        # Cache the compiled kernel to avoid per-call JIT dispatch overhead.
        self._compiled = self.kernel(self.config["threads"])

    @property
    def default_config(self) -> dict:
        return {"threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": t} for t in [64, 128, 256, 512]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._compiled(x)
