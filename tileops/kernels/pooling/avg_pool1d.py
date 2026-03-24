"""Average Pooling 1D Forward Kernel using TileLang.

Applies average pooling over an input signal composed of several input planes.
The input is 3D tensor (batch, channels, in_seq_len).
The output is 3D tensor (batch, channels, out_seq_len).
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["AvgPooling1dKernel"]


def _calculate_out_size(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    """Calculate output size for pooling."""
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@functools.lru_cache(maxsize=32)
def _avg_pooling_1d_fwd_kernel(
    batch: int,
    channels: int,
    in_seq_len: int,
    out_seq_len: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    @tilelang.jit(out_idx=[1])
    def _func(bdim: int, threads: int) -> None:
        @T.prim_func
        def main(
            x: T.Tensor((batch, channels, in_seq_len), dtype),
            y: T.Tensor((batch, channels, out_seq_len), dtype),
        ) -> None:
            # Tile over all output positions with bdim threads per block
            with T.Kernel(batch * channels, out_seq_len, threads=threads) as (i_bc, i_o):
                i_b = i_bc // channels
                i_c = i_bc % channels

                # Allocate local accumulator
                acc = T.alloc_fragment((1,), accum_dtype)
                if i_o < out_seq_len:
                    acc[0] = T.cast(0, accum_dtype)
                    # Sum over kernel window
                    for ki in T.Serial(kernel_size):
                        in_idx = i_o * stride - padding + ki * dilation
                        if 0 <= in_idx < in_seq_len:
                            val = T.cast(x[i_b, i_c, in_idx], accum_dtype)
                            acc[0] = acc[0] + val
                    # Write average
                    result = T.cast(acc[0] / T.cast(kernel_size, accum_dtype), dtype)
                    y[i_b, i_c, i_o] = result

        return main

    return _func


@torch.library.custom_op("top::avg_pooling_1d_fwd", mutates_args=())
def _avg_pooling_1d_fwd_wrapped(
    batch: int,
    channels: int,
    in_seq_len: int,
    out_seq_len: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype_str: str,
    accum_dtype_str: str,
    bdim: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _avg_pooling_1d_fwd_kernel(
        batch,
        channels,
        in_seq_len,
        out_seq_len,
        kernel_size,
        stride,
        padding,
        dilation,
        dtype_str,
        accum_dtype_str,
    )(bdim, threads)(x)


@_avg_pooling_1d_fwd_wrapped.register_fake
def _(
    batch: int,
    channels: int,
    in_seq_len: int,
    out_seq_len: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype_str: str,
    accum_dtype_str: str,
    bdim: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (in_seq_len, kernel_size, stride, padding, dilation, bdim, threads, dtype_str, accum_dtype_str)
    return torch.empty((batch, channels, out_seq_len), dtype=x.dtype, device=x.device)


class AvgPooling1dKernel(Kernel):
    """Average Pooling 1D Forward Kernel.

    Supports SM80+ architectures. Applies average pooling with configurable
    kernel_size, stride, padding, and dilation.

    Args:
        batch: Batch size.
        channels: Number of channels.
        in_seq_len: Input sequence length.
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Padding added to input. If None, defaults to 0.
        dilation: Dilation of the pooling window. If None, defaults to 1.
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
        in_seq_len: int,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: Optional[int] = None,
        dilation: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.channels = channels
        self.in_seq_len = in_seq_len
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding if padding is not None else 0
        self.dilation = dilation if dilation is not None else 1
        self.dtype = dtype
        self.accum_dtype = accum_dtype

        # Calculate output sequence length
        self.out_seq_len = _calculate_out_size(
            self.in_seq_len, self.kernel_size, self.stride, self.padding, self.dilation
        )

        self.kernel = _avg_pooling_1d_fwd_kernel(
            self.batch,
            self.channels,
            self.in_seq_len,
            self.out_seq_len,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
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
        threads = [128, 256]
        bdims = [1]
        return [{"bdim": b, "threads": t} for b in bdims for t in threads]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _avg_pooling_1d_fwd_wrapped(
            self.batch,
            self.channels,
            self.in_seq_len,
            self.out_seq_len,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.dtype_str,
            self.accum_dtype_str,
            self.config["bdim"],
            self.config["threads"],
            x,
        )
