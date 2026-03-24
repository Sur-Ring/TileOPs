"""Benchmarks for MaxPoolingFwdOp."""

import itertools

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkFixture, BenchmarkReport
from tests.ops.test_max_pool2d import MaxPoolingFwdFixture
from tileops.ops.pooling import MaxPoolingFwdOp


class MaxPoolingFwdBenchmark(BenchmarkBase):
    def calculate_flops(self, *args, **kwargs) -> int:
        """Calculate FLOPs for max pooling.

        FLOPs = batch * out_seq_len * dim * kernel_size
        (comparison operations)
        """
        batch = kwargs.get("batch", 1)
        out_seq_len = kwargs.get("out_seq_len", 1)
        dim = kwargs.get("dim", 1)
        kernel_size = kwargs.get("kernel_size", 1)
        return batch * out_seq_len * dim * (kernel_size - 1)

    def calculate_memory(self, *args, **kwargs) -> int:
        """Calculate memory traffic.

        Read: batch * in_seq_len * dim
        Write: batch * out_seq_len * dim
        """
        batch = kwargs.get("batch", 1)
        in_seq_len = kwargs.get("in_seq_len", 1)
        dim = kwargs.get("dim", 1)
        out_seq_len = kwargs.get("out_seq_len", 1)

        bytes_per_element = 2  # fp16/bf16
        read = batch * in_seq_len * dim * bytes_per_element
        written = batch * out_seq_len * dim * bytes_per_element
        return read + written


class MaxPoolingFwdBenchmarkFixture(BenchmarkFixture):
    PARAMS = list(
        itertools.product(
            [1, 2],  # batch
            [256, 512, 1024],  # in_seq_len
            [2048, 4096],  # dim
            [2, 4],  # kernel_size
            [torch.float16, torch.bfloat16],
        )
    )


@MaxPoolingFwdBenchmarkFixture
def bench_max_pooling_fwd(
    batch: int,
    in_seq_len: int,
    dim: int,
    kernel_size: int,
    dtype: torch.dtype,
) -> None:
    test = MaxPoolingFwdFixture()
    test_instance = type("TestInstance", (), {
        "batch": batch,
        "in_seq_len": in_seq_len,
        "dim": dim,
        "kernel_size": kernel_size,
        "stride": kernel_size,
        "padding": 0,
        "dilation": 1,
        "dtype": dtype,
    })()

    op = MaxPoolingFwdOp(
        kernel_size=kernel_size,
        stride=kernel_size,
        padding=0,
        dilation=1,
        dtype=dtype,
    )

    bm = MaxPoolingFwdBenchmark(test_instance)
    inputs = (torch.randn(batch, in_seq_len, dim, dtype=dtype, device="cuda"),)

    # Calculate output shape
    out_seq_len = (in_seq_len + 2 * 0 - 1 * (kernel_size - 1) - 1) // kernel_size + 1

    result = bm.profile(
        op,
        *inputs,
        batch=batch,
        in_seq_len=in_seq_len,
        dim=dim,
        out_seq_len=out_seq_len,
        kernel_size=kernel_size,
    )

    BenchmarkReport.record(
        op,
        locals(),
        result,
        tag="tileops",
        batch=batch,
        in_seq_len=in_seq_len,
        dim=dim,
        out_seq_len=out_seq_len,
        kernel_size=kernel_size,
        dtype=str(dtype),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs", "-k", "bench"])
