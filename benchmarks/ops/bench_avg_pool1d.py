"""Benchmarks for AvgPooling1dOp."""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import AvgPooling1dOp


class AvgPooling1dBenchmark(BenchmarkBase):
    """Benchmark for AvgPooling1dOp."""

    def __init__(self, test):
        super().__init__(test)
        self.batch = test.batch
        self.channels = test.channels
        self.in_seq_len = test.in_seq_len
        self.out_seq_len = test.out_seq_len
        self.kernel_size = test.kernel_size

    def calculate_flops(self) -> Optional[float]:
        """Calculate FLOPs for avg pooling 1D.

        Avg pooling performs kernel_size additions + 1 division per output element.
        """
        return self.batch * self.channels * self.out_seq_len * (self.kernel_size + 1)

    def calculate_memory(self) -> Optional[float]:
        """Calculate memory traffic.

        Read: batch * channels * in_seq_len
        Write: batch * channels * out_seq_len
        """
        bytes_per_element = 2  # fp16/bf16
        read = self.batch * self.channels * self.in_seq_len * bytes_per_element
        written = self.batch * self.channels * self.out_seq_len * bytes_per_element
        return read + written


def _calculate_out_size(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    """Calculate output size for pooling."""
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


_AVG_POOL1D_BENCH_PARAMS = [
    # Small shape
    pytest.param(1, 3, 100, 2, 2, 0, 1, torch.float16, id="small-100"),
    # Medium shape
    pytest.param(1, 3, 512, 2, 2, 0, 1, torch.float16, id="medium-512"),
    pytest.param(1, 3, 1024, 2, 2, 0, 1, torch.float16, id="large-1024"),
    # Large shape with more channels
    pytest.param(1, 64, 512, 2, 2, 0, 1, torch.float16, id="deep-64x512"),
    # Different stride
    pytest.param(1, 16, 256, 4, 4, 0, 1, torch.float16, id="stride-4"),
    # With padding
    pytest.param(1, 3, 512, 3, 2, 1, 1, torch.float16, id="with-padding"),
    # BF16
    pytest.param(1, 3, 512, 2, 2, 0, 1, torch.bfloat16, id="bf16"),
    # Batched
    pytest.param(2, 16, 256, 3, 1, 1, 1, torch.float16, id="batched"),
    # Dilation
    pytest.param(1, 4, 128, 3, 1, 2, 2, torch.float16, id="dilation"),
]


@pytest.mark.parametrize(
    "batch, channels, in_seq_len, kernel_size, stride, padding, dilation, dtype",
    _AVG_POOL1D_BENCH_PARAMS,
)
def bench_avg_pool1d(
    batch: int,
    channels: int,
    in_seq_len: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: torch.dtype,
) -> None:
    """Benchmark AvgPooling1dOp."""
    # Calculate output dimensions
    out_seq_len = _calculate_out_size(in_seq_len, kernel_size, stride, padding, dilation)

    # Create test instance for benchmark calculations
    class TestInstance:
        pass

    test_instance = TestInstance()
    test_instance.batch = batch
    test_instance.channels = channels
    test_instance.in_seq_len = in_seq_len
    test_instance.out_seq_len = out_seq_len
    test_instance.kernel_size = kernel_size

    op = AvgPooling1dOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
    )

    # Generate input
    x = torch.randn(batch, channels, in_seq_len, dtype=dtype, device="cuda")

    bm = AvgPooling1dBenchmark(test_instance)

    # Profile tileops
    result = bm.profile(op, x)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Profile baseline (PyTorch)
    def baseline_fn():
        return torch.nn.functional.avg_pool1d(
            x,
            kernel_size=kernel_size,
            stride=stride if stride else kernel_size,
            padding=padding if padding else 0,
            dilation=dilation if dilation else 1,
        )

    result_bl = bm.profile(baseline_fn)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
