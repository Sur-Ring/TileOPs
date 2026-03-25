"""Average Pooling 3D Forward Kernel using TileLang.

Applies average pooling over an input signal composed of several input planes.
The input is 5D tensor (batch, channels, depth, height, width).
The output is 5D tensor (batch, channels, out_d, out_h, out_w).
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["AvgPooling3dKernel"]


def _calculate_out_size(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    """Calculate output size for pooling."""
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@functools.lru_cache(maxsize=32)
def _avg_pooling_3d_fwd_kernel(
    batch: int,
    channels: int,
    in_d: int,
    in_h: int,
    in_w: int,
    out_d: int,
    out_h: int,
    out_w: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    padding_d: int,
    padding_h: int,
    padding_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    @tilelang.jit(out_idx=[1])
    def _func(bdim: int, threads: int) -> None:
        @T.prim_func
        def main(
            x: T.Tensor((batch, channels, in_d, in_h, in_w), dtype),
            y: T.Tensor((batch, channels, out_d, out_h, out_w), dtype),
        ) -> None:
            # Restructure to 3D grid: (batch*channels, out_d*out_h, out_w)
            # i_dhw will be split into i_d and i_h within the kernel
            with T.Kernel(batch * channels, out_d * out_h, out_w, threads=threads) as (i_bc, i_dhw, i_w):
                i_b = i_bc // channels
                i_c = i_bc % channels
                i_d = i_dhw // out_h
                i_h = i_dhw % out_h

                # Allocate local accumulator
                acc = T.alloc_fragment((1,), accum_dtype)
                if i_d < out_d and i_h < out_h and i_w < out_w:
                    acc[0] = T.cast(0, accum_dtype)
                    # Sum over kernel window
                    for kdi in T.Serial(kernel_d):
                        for khi in T.Serial(kernel_h):
                            for kwi in T.Serial(kernel_w):
                                d_idx = i_d * stride_d - padding_d + kdi * dilation_d
                                h_idx = i_h * stride_h - padding_h + khi * dilation_h
                                w_idx = i_w * stride_w - padding_w + kwi * dilation_w
                                if 0 <= d_idx < in_d and 0 <= h_idx < in_h and 0 <= w_idx < in_w:
                                    val = T.cast(x[i_b, i_c, d_idx, h_idx, w_idx], accum_dtype)
                                    acc[0] = acc[0] + val
                    # Write average
                    kernel_size_sum = kernel_d * kernel_h * kernel_w
                    result = T.cast(acc[0] / T.cast(kernel_size_sum, accum_dtype), dtype)
                    y[i_b, i_c, i_d, i_h, i_w] = result

        return main

    return _func


@torch.library.custom_op("top::avg_pooling_3d_fwd", mutates_args=())
def _avg_pooling_3d_fwd_wrapped(
    batch: int,
    channels: int,
    in_d: int,
    in_h: int,
    in_w: int,
    out_d: int,
    out_h: int,
    out_w: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    padding_d: int,
    padding_h: int,
    padding_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    dtype_str: str,
    accum_dtype_str: str,
    bdim: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _avg_pooling_3d_fwd_kernel(
        batch,
        channels,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
        dtype_str,
        accum_dtype_str,
    )(bdim, threads)(x)


@_avg_pooling_3d_fwd_wrapped.register_fake
def _(
    batch: int,
    channels: int,
    in_d: int,
    in_h: int,
    in_w: int,
    out_d: int,
    out_h: int,
    out_w: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    padding_d: int,
    padding_h: int,
    padding_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    dtype_str: str,
    accum_dtype_str: str,
    bdim: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (
        in_d, in_h, in_w, kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        bdim, threads, dtype_str, accum_dtype_str
    )
    return torch.empty((batch, channels, out_d, out_h, out_w), dtype=x.dtype, device=x.device)


class AvgPooling3dKernel(Kernel):
    """Average Pooling 3D Forward Kernel.

    Supports SM80+ architectures. Applies average pooling with configurable
    kernel_size, stride, padding, and dilation.

    Args:
        batch: Batch size.
        channels: Number of channels.
        in_d: Input depth.
        in_h: Input height.
        in_w: Input width.
        kernel_size: Size of the pooling window (d, h, w) or a single int for cubic.
        stride: Stride of the pooling window (d, h, w) or a single int. If None, defaults to kernel_size.
        padding: Padding added to input (d, h, w) or a single int. If None, defaults to 0.
        dilation: Dilation of the pooling window (d, h, w) or a single int. If None, defaults to 1.
        dtype: Data type (float16 or bfloat16).
        accum_dtype: Accumulator dtype for sum reduction (float32 recommended).
        config: Optional kernel configuration.
        tune: Whether to run autotuning.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        channels: int,
        in_d: int,
        in_h: int,
        in_w: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] | None = None,
        padding: int | tuple[int, int, int] | None = None,
        dilation: int | tuple[int, int, int] | None = None,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.channels = channels
        self.in_d = in_d
        self.in_h = in_h
        self.in_w = in_w

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_d = kernel_size
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_d, self.kernel_h, self.kernel_w = kernel_size

        # Handle stride
        if stride is None:
            self.stride_d = self.kernel_d
            self.stride_h = self.kernel_h
            self.stride_w = self.kernel_w
        elif isinstance(stride, int):
            self.stride_d = stride
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_d, self.stride_h, self.stride_w = stride

        # Handle padding
        if padding is None:
            self.padding_d = 0
            self.padding_h = 0
            self.padding_w = 0
        elif isinstance(padding, int):
            self.padding_d = padding
            self.padding_h = padding
            self.padding_w = padding
        else:
            self.padding_d, self.padding_h, self.padding_w = padding

        # Handle dilation
        if dilation is None:
            self.dilation_d = 1
            self.dilation_h = 1
            self.dilation_w = 1
        elif isinstance(dilation, int):
            self.dilation_d = dilation
            self.dilation_h = dilation
            self.dilation_w = dilation
        else:
            self.dilation_d, self.dilation_h, self.dilation_w = dilation

        self.dtype = dtype
        self.accum_dtype = accum_dtype

        # Calculate output dimensions
        self.out_d = _calculate_out_size(
            self.in_d, self.kernel_d, self.stride_d, self.padding_d, self.dilation_d
        )
        self.out_h = _calculate_out_size(
            self.in_h, self.kernel_h, self.stride_h, self.padding_h, self.dilation_h
        )
        self.out_w = _calculate_out_size(
            self.in_w, self.kernel_w, self.stride_w, self.padding_w, self.dilation_w
        )

        self.kernel = _avg_pooling_3d_fwd_kernel(
            self.batch,
            self.channels,
            self.in_d,
            self.in_h,
            self.in_w,
            self.out_d,
            self.out_h,
            self.out_w,
            self.kernel_d,
            self.kernel_h,
            self.kernel_w,
            self.stride_d,
            self.stride_h,
            self.stride_w,
            self.padding_d,
            self.padding_h,
            self.padding_w,
            self.dilation_d,
            self.dilation_h,
            self.dilation_w,
            self.dtype_str,
            self.accum_dtype_str,
        )
        self.init_config(config, tune)

    @property
    def dtype_str(self) -> str:
        """Convert dtype to str for tl kernels"""
        return self.dtype_to_str(self.dtype)

    @property
    def accum_dtype_str(self) -> str:
        """Convert accum_dtype to str for tl kernels"""
        return self.dtype_to_str(self.accum_dtype)

    @staticmethod
    def dtype_to_str(dtype: torch.dtype) -> str:
        """Convert a torch dtype to the TileLang dtype string."""
        return str(dtype).split('.')[-1]

    @property
    def default_config(self) -> dict:
        return {"bdim": 1, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        bdims = [1]
        threads_list = [128, 256]
        return [{"bdim": b, "threads": t} for b in bdims for t in threads_list]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _avg_pooling_3d_fwd_wrapped(
            self.batch,
            self.channels,
            self.in_d,
            self.in_h,
            self.in_w,
            self.out_d,
            self.out_h,
            self.out_w,
            self.kernel_d,
            self.kernel_h,
            self.kernel_w,
            self.stride_d,
            self.stride_h,
            self.stride_w,
            self.padding_d,
            self.padding_h,
            self.padding_w,
            self.dilation_d,
            self.dilation_h,
            self.dilation_w,
            self.dtype_str,
            self.accum_dtype_str,
            self.config["bdim"],
            self.config["threads"],
            x,
        )
