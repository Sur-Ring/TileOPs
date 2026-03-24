"""Benchmarks for AvgPooling3dOp."""

from typing import Optional, Union

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import AvgPooling3dOp


class AvgPooling3dBenchmark(BenchmarkBase):
    """Benchmark for AvgPooling3dOp."""

    def __init__(self, test):
        super().__init__(test)
        self.batch = test.batch
        self.channels = test.channels
        self.in_d = test.in_d
        self.in_h = test.in_h
        self.in_w = test.in_w
        self.out_d = test.out_d
        self.out_h = test.out_h
        self.out_w = test.out_w
        self.kernel_size = test.kernel_size

    def calculate_flops(self) -> Optional[float]:
        """Calculate FLOPs for avg pooling 3D.

        Avg pooling performs kernel_size_d * kernel_size_h * kernel_size_w additions + 1 division per output element.
        """
        if isinstance(self.kernel_size, int):
            kd = kh = kw = self.kernel_size
        else:
            kd, kh, kw = self.kernel_size
        return self.batch * self.channels * self.out_d * self.out_h * self.out_w * (kd * kh * kw + 1)

    def calculate_memory(self) -> Optional[float]:
        """Calculate memory traffic.

        Read: batch * channels * in_d * in_h * in_w
        Write: batch * channels * out_d * out_h * out_w
        """
        bytes_per_element = 2  # fp16/bf16
        read = self.batch * self.channels * self.in_d * self.in_h * self.in_w * bytes_per_element
        written = self.batch * self.channels * self.out_d * self.out_h * self.out_w * bytes_per_element
        return read + written


def _calculate_out_size(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    """Calculate output size for pooling."""
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


_AVG_POOL3D_BENCH_PARAMS = [
    # Small shape
    pytest.param(1, 3, 4, 16, 16, 2, 2, 0, 1, torch.float16, id="small-16x16"),
    # Medium shape
    pytest.param(1, 3, 8, 32, 32, 2, 2, 0, 1, torch.float16, id="medium-32x32"),
    pytest.param(1, 3, 8, 64, 64, 2, 2, 0, 1, torch.float16, id="large-64x64"),
    # Large shape with more channels
    pytest.param(1, 16, 4, 32, 32, 2, 2, 0, 1, torch.float16, id="deep-16x32x32"),
    # Rectangular kernel
    pytest.param(1, 3, 8, 32, 32, (2, 2, 2), 2, 0, 1, torch.float16, id="cubic"),
    # Different stride
    pytest.param(1, 4, 8, 32, 32, 2, 2, 0, 1, torch.float16, id="stride-2"),
    # BF16
    pytest.param(1, 3, 8, 32, 32, 2, 2, 0, 1, torch.bfloat16, id="bf16"),
    # Batched
    pytest.param(2, 8, 4, 16, 16, 2, 2, 0, 1, torch.float16, id="batched"),
]


@pytest.mark.parametrize(
    "batch, channels, in_d, in_h, in_w, kernel_size, stride, padding, dilation, dtype",
    _AVG_POOL3D_BENCH_PARAMS,
)
def bench_avg_pool3d(
    batch: int,
    channels: int,
    in_d: int,
    in_h: int,
    in_w: int,
    kernel_size: Union[int, tuple[int, int, int]],
    stride: Union[int, tuple[int, int, int]],
    padding: Union[int, tuple[int, int, int]],
    dilation: Union[int, tuple[int, int, int]],
    dtype: torch.dtype,
) -> None:
    """Benchmark AvgPooling3dOp."""
    # Handle kernel_size
    if isinstance(kernel_size, int):
        kd = kh = kw = kernel_size
    else:
        kd, kh, kw = kernel_size

    # Handle stride
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    # Handle padding
    if isinstance(padding, int):
        padding_d = padding_h = padding_w = padding
    else:
        padding_d, padding_h, padding_w = padding

    # Handle dilation
    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    # Calculate output dimensions
    out_d = _calculate_out_size(in_d, kd, stride_d, padding_d, dilation_d)
    out_h = _calculate_out_size(in_h, kh, stride_h, padding_h, dilation_h)
    out_w = _calculate_out_size(in_w, kw, stride_w, padding_w, dilation_w)

    # Create test instance for benchmark calculations
    class TestInstance:
        pass

    test_instance = TestInstance()
    test_instance.batch = batch
    test_instance.channels = channels
    test_instance.in_d = in_d
    test_instance.in_h = in_h
    test_instance.in_w = in_w
    test_instance.out_d = out_d
    test_instance.out_h = out_h
    test_instance.out_w = out_w
    test_instance.kernel_size = kernel_size

    op = AvgPooling3dOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
    )

    # Generate input
    x = torch.randn(batch, channels, in_d, in_h, in_w, dtype=dtype, device="cuda")

    bm = AvgPooling3dBenchmark(test_instance)

    # Profile tileops
    result = bm.profile(op, x)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Profile baseline (PyTorch)
    def baseline_fn():
        return torch.nn.functional.avg_pool3d(
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
