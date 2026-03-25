"""Max Pooling 2D Forward Kernel using TileLang.

Applies max pooling over an input signal composed of several input planes.
The input is 4D tensor (batch, channels, height, width).
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["MaxPooling2dFwdKernel"]


def _calculate_out_size(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    """Calculate output size for pooling."""
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@functools.lru_cache(maxsize=32)
def _max_pooling_2d_fwd_kernel(
    batch: int,
    channels: int,
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
    dilation_h: int,
    dilation_w: int,
    dtype: str,
) -> Callable:
    # Flat total number of output elements processed by the kernel.
    total = batch * channels * out_h * out_w

    @tilelang.jit(out_idx=[1])
    def _func(threads: int) -> None:
        @T.prim_func
        def main(
            x: T.Tensor((batch, channels, in_h, in_w), dtype),
            y: T.Tensor((batch, channels, out_h, out_w), dtype),
        ) -> None:
            # Flat 1-D grid: each block processes `threads` output elements.
            # Single-pass: write directly to y[b,c,oh,ow] inside the Serial
            # loop, eliminating the intermediate fragment and the second
            # T.Parallel write-back pass.
            with T.Kernel(T.ceildiv(total, threads), threads=threads) as pid:
                for i in T.Parallel(threads):
                    lin = pid * threads + i
                    if lin < total:
                        ow = lin % out_w
                        oh = (lin // out_w) % out_h
                        c  = (lin // (out_w * out_h)) % channels
                        b  = lin // (out_w * out_h * channels)
                        y[b, c, oh, ow] = T.cast(-float("inf"), dtype)
                        for kh in T.Serial(kernel_h):
                            for kw in T.Serial(kernel_w):
                                h_idx = oh * stride_h - padding_h + kh * dilation_h
                                w_idx = ow * stride_w - padding_w + kw * dilation_w
                                if 0 <= h_idx < in_h and 0 <= w_idx < in_w:
                                    y[b, c, oh, ow] = T.max(y[b, c, oh, ow], x[b, c, h_idx, w_idx])

        return main

    return _func


class MaxPooling2dFwdKernel(Kernel):
    """Max Pooling 2D Forward Kernel.

    Supports SM80+ architectures. Applies max pooling with configurable
    kernel_size, stride, padding, and dilation.

    Args:
        batch: Batch size.
        channels: Number of channels.
        in_h: Input height.
        in_w: Input width.
        kernel_size: Size of the pooling window (h, w) or a single int for square.
        stride: Stride of the pooling window (h, w) or a single int. If None, defaults to kernel_size.
        padding: Padding added to input (h, w) or a single int. If None, defaults to 0.
        dilation: Dilation of the pooling window (h, w) or a single int. If None, defaults to 1.
        dtype: Data type (float16 or bfloat16).
        config: Optional kernel configuration.
        tune: Whether to run autotuning.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        channels: int,
        in_h: int,
        in_w: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] | None = None,
        dilation: int | tuple[int, int] | None = None,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.channels = channels
        self.in_h = in_h
        self.in_w = in_w

        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        if stride is None:
            self.stride_h = self.kernel_h
            self.stride_w = self.kernel_w
        elif isinstance(stride, int):
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride

        if padding is None:
            self.padding_h = 0
            self.padding_w = 0
        elif isinstance(padding, int):
            self.padding_h = padding
            self.padding_w = padding
        else:
            self.padding_h, self.padding_w = padding

        if dilation is None:
            self.dilation_h = 1
            self.dilation_w = 1
        elif isinstance(dilation, int):
            self.dilation_h = dilation
            self.dilation_w = dilation
        else:
            self.dilation_h, self.dilation_w = dilation

        self.dtype = dtype

        self.out_h = _calculate_out_size(
            self.in_h, self.kernel_h, self.stride_h, self.padding_h, self.dilation_h
        )
        self.out_w = _calculate_out_size(
            self.in_w, self.kernel_w, self.stride_w, self.padding_w, self.dilation_w
        )

        self.kernel = _max_pooling_2d_fwd_kernel(
            self.batch,
            self.channels,
            self.in_h,
            self.in_w,
            self.out_h,
            self.out_w,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
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
