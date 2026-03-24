"""Max Pooling 1D Forward Kernel using TileLang.

Applies max pooling over an input signal composed of several input planes.
The input is 3D tensor (batch, in_seq_len, dim).
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
    in_seq_len: int,
    dim: int,
    out_seq_len: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: str,
) -> Callable:
    @tilelang.jit(out_idx=[3])
    def _func(block_b: int, threads: int) -> None:
        @T.prim_func
        def main(
            x: T.Tensor((batch, in_seq_len, dim), dtype),
            y: T.Tensor((batch, out_seq_len, dim), dtype),
        ) -> None:
            with T.Kernel(
                T.ceildiv(batch, block_b), dim, out_seq_len, threads=threads
            ) as (pid_b, pid_d, pid_o):
                b = pid_b * block_b
                block_size = T.min(block_b, batch - b)

                # Allocate shared and local buffers
                ks = kernel_size
                x_shared = T.alloc_shared((block_size, ks, dim), dtype)
                x_local = T.alloc_fragment((block_size, ks, dim), dtype)
                out_local = T.alloc_fragment((block_size, dim), dtype)

                # Initialize output to very small value
                for i, j in T.Parallel(block_size, dim):
                    out_local[i, j] = T.cast(-float("inf"), dtype)

                # Load input data for this output position
                for ti in T.Parallel(block_size, ks):
                    in_idx = pid_o * stride - padding + ti * dilation
                    if 0 <= in_idx < in_seq_len:
                        T.copy(x[b + ti, in_idx, pid_d], x_shared[ti, pid_d])

                T.copy(x_shared, x_local)

                # Compute max over kernel
                for ki in T.Serial(ks):
                    for i, j in T.Parallel(block_size, dim):
                        out_local[i, j] = T.max(out_local[i, j], x_local[ki, i, j])

                # Store output
                for i, j in T.Parallel(block_size, dim):
                    y[b + i, pid_o, j] = out_local[i, j]

        return main

    return _func


@torch.library.custom_op("top::max_pooling_1d_fwd", mutates_args=())
def _max_pooling_1d_fwd_wrapped(
    batch: int,
    in_seq_len: int,
    dim: int,
    out_seq_len: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype_str: str,
    block_b: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _max_pooling_1d_fwd_kernel(
        batch,
        in_seq_len,
        dim,
        out_seq_len,
        kernel_size,
        stride,
        padding,
        dilation,
        dtype_str,
    )(block_b, threads)(x)


@_max_pooling_1d_fwd_wrapped.register_fake
def _(
    batch: int,
    in_seq_len: int,
    dim: int,
    out_seq_len: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype_str: str,
    block_b: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (in_seq_len, kernel_size, stride, padding, dilation, block_b, threads, dtype_str)
    return torch.empty((batch, out_seq_len, dim), dtype=x.dtype, device=x.device)


class MaxPooling1dFwdKernel(Kernel):
    """Max Pooling 1D Forward Kernel.

    Supports SM80+ architectures. Applies max pooling with configurable
    kernel_size, stride, padding, and dilation.

    Args:
        batch: Batch size.
        in_seq_len: Input sequence length.
        dim: Feature dimension.
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
        in_seq_len: int,
        dim: int,
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
        self.in_seq_len = in_seq_len
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding if padding is not None else 0
        self.dilation = dilation if dilation is not None else 1
        self.dtype = dtype

        # Calculate output sequence length
        self.out_seq_len = _calculate_out_size(
            self.in_seq_len, self.kernel_size, self.stride, self.padding, self.dilation
        )

        self.kernel = _max_pooling_1d_fwd_kernel(
            self.batch,
            self.in_seq_len,
            self.dim,
            self.out_seq_len,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
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
        return _max_pooling_1d_fwd_wrapped(
            self.batch,
            self.in_seq_len,
            self.dim,
            self.out_seq_len,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.dtype_str,
            self.config["block_b"],
            self.config["threads"],
            x,
        )