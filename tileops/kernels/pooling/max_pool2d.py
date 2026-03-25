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
    @tilelang.jit(out_idx=[1])
    def _func(block_ow: int, threads: int) -> None:
        @T.prim_func
        def main(
            x: T.Tensor((batch, channels, in_h, in_w), dtype),
            y: T.Tensor((batch, channels, out_h, out_w), dtype),
        ) -> None:
            # 2D grid: (batch * channels * out_h, ceildiv(out_w, block_ow))
            # Each block processes block_ow consecutive out_w positions for one (b, c, oh).
            # Thread utilization: block_ow / threads ≈ (threads-1)/threads ≈ 100%
            # (no shared memory; each thread independently traverses kernel window in registers)
            with T.Kernel(
                batch * channels * out_h,
                T.ceildiv(out_w, block_ow),
                threads=threads,
            ) as (pid_bch, pid_ow_blk):
                # Decode (b, c, oh) from pid_bch
                b = pid_bch // (channels * out_h)
                temp = pid_bch % (channels * out_h)
                c = temp // out_h
                pid_oh = temp % out_h
                ow_start = pid_ow_blk * block_ow

                # Each thread independently accumulates max in a register fragment.
                # Serial kernel-window traversal avoids shared memory entirely.
                out_local = T.alloc_fragment((block_ow,), dtype)

                for i in T.Parallel(block_ow):
                    out_local[i] = T.cast(-float("inf"), dtype)

                for kh_idx in T.Serial(kernel_h):
                    for kw_idx in T.Serial(kernel_w):
                        for i in T.Parallel(block_ow):
                            ow = ow_start + i
                            h_idx = pid_oh * stride_h - padding_h + kh_idx * dilation_h
                            w_idx = ow * stride_w - padding_w + kw_idx * dilation_w
                            if ow < out_w and 0 <= h_idx < in_h and 0 <= w_idx < in_w:
                                out_local[i] = T.max(out_local[i], x[b, c, h_idx, w_idx])

                # Write results to global memory (one pass, boundary-guarded)
                for i in T.Parallel(block_ow):
                    ow = ow_start + i
                    if ow < out_w:
                        y[b, c, pid_oh, ow] = out_local[i]

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
    block_ow: int,
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
    )(block_ow, threads)(x)


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
    block_ow: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (
        in_h, in_w, kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w, block_ow, threads, dtype_str
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
        # Cache the compiled kernel (JIT + config) to avoid per-call compilation overhead.
        self._compiled = self.kernel(self.config["block_ow"], self.config["threads"])

    @property
    def default_config(self) -> dict:
        # block_ow = threads - 1 maximises thread utilisation (~99.6%).
        # No shared memory means the only constraint is block_ow < threads.
        return {"block_ow": 255, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for threads in [64, 128, 256, 512]:
            for block_ow in [threads - 1, threads // 2, threads // 4]:
                if block_ow > 0:
                    configs.append({"block_ow": block_ow, "threads": threads})
        return configs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._compiled(x)
