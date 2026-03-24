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
    @tilelang.jit(out_idx=[4])
    def _func(block_b: int, threads: int) -> None:
        @T.prim_func
        def main(
            x: T.Tensor((batch, channels, in_h, in_w), dtype),
            y: T.Tensor((batch, channels, out_h, out_w), dtype),
        ) -> None:
            with T.Kernel(
                T.ceildiv(batch, block_b), channels, out_h, out_w, threads=threads
            ) as (pid_b, pid_c, pid_h, pid_w):
                b = pid_b * block_b
                block_size = T.min(block_b, batch - b)

                # Allocate shared memory for kernel window
                kh = kernel_h
                kw = kernel_w
                x_shared = T.alloc_shared((kh, kw, block_size, in_w), dtype)
                out_local = T.alloc_fragment((block_size, in_w), dtype)

                # Initialize output to very small value
                for i, j in T.Parallel(block_size, in_w):
                    out_local[i, j] = T.cast(-float("inf"), dtype)

                # Load input data for this output position
                for khi, kwi in T.Parallel(kh, kw):
                    h_idx = pid_h * stride_h - padding_h + khi * dilation_h
                    w_idx = pid_w * stride_w - padding_w + kwi * dilation_w
                    if 0 <= h_idx < in_h and 0 <= w_idx < in_w:
                        for i in T.Parallel(block_size):
                            T.copy(x[b + i, pid_c, h_idx, w_idx], x_shared[khi, kwi, i])

                # Compute max over kernel
                for khi in T.Serial(kh):
                    for kwi in T.Serial(kw):
                        for i, j in T.Parallel(block_size, in_w):
                            out_local[i, j] = T.max(out_local[i, j], x_shared[khi, kwi, i, j])

                # Store output
                for i, j in T.Parallel(block_size, in_w):
                    y[b + i, pid_c, pid_h, pid_w] = out_local[i, j]

        return main

    return _func


@torch.library.custom_op("top::max_pooling_2d_fwd", mutates_args=())
def _max_pooling_2d_fwd_wrapped(
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
    dtype_str: str,
    block_b: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _max_pooling_2d_fwd_kernel(
        batch,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        dtype_str,
    )(block_b, threads)(x)


@_max_pooling_2d_fwd_wrapped.register_fake
def _(
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
    dtype_str: str,
    block_b: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (
        in_h, in_w, kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w, block_b, threads, dtype_str
    )
    return torch.empty((batch, channels, out_h, out_w), dtype=x.dtype, device=x.device)


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

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        # Handle stride
        if stride is None:
            self.stride_h = self.kernel_h
            self.stride_w = self.kernel_w
        elif isinstance(stride, int):
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride

        # Handle padding
        if padding is None:
            self.padding_h = 0
            self.padding_w = 0
        elif isinstance(padding, int):
            self.padding_h = padding
            self.padding_w = padding
        else:
            self.padding_h, self.padding_w = padding

        # Handle dilation
        if dilation is None:
            self.dilation_h = 1
            self.dilation_w = 1
        elif isinstance(dilation, int):
            self.dilation_h = dilation
            self.dilation_w = dilation
        else:
            self.dilation_h, self.dilation_w = dilation

        self.dtype = dtype

        # Calculate output dimensions
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

    @property
    def default_config(self) -> dict:
        return {"block_b": 1, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        block_bs = [1, 2, 4]
        threads_list = [64, 128, 256]
        return [{"block_b": bb, "threads": t} for bb in block_bs for t in threads_list]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _max_pooling_2d_fwd_wrapped(
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
            self.config["block_b"],
            self.config["threads"],
            x,
        )