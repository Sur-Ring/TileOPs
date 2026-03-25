"""Benchmarks for MaxPooling1dFwdOp."""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MaxPooling1dFwdOp


class MaxPooling1dBenchmark(BenchmarkBase):
    """Benchmark for MaxPooling1dFwdOp."""

    def __init__(self, test):
        super().__init__(test)
        self.batch = test.batch
        self.channels = test.channels
        self.in_length = test.in_length
        self.out_length = test.out_length
        self.kernel_size = test.kernel_size
        self.stride = test.stride
        self.padding = test.padding
        self.dilation = test.dilation

    def calculate_flops(self) -> Optional[float]:
        """Max pooling performs (kernel_size - 1) comparisons per output element."""
        return self.batch * self.channels * self.out_length * (self.kernel_size - 1)

    def calculate_memory(self) -> Optional[float]:
        """Read input + write output (fp16/bf16 = 2 bytes each)."""
        bpe = 2
        read = self.batch * self.channels * self.in_length * bpe
        write = self.batch * self.channels * self.out_length * bpe
        return read + write


def _out(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


_MAX_POOL1D_BENCH_PARAMS = [
    # -----------------------------------------------------------------------
    # Common 1D pooling shapes (audio, time-series, 1D convnets)
    # -----------------------------------------------------------------------
    pytest.param(1, 64, 512, 3, 2, 1, 1, torch.float16,  id="c64-l512-k3-fp16"),
    pytest.param(1, 64, 512, 3, 2, 1, 1, torch.bfloat16, id="c64-l512-k3-bf16"),
    pytest.param(4, 64, 512, 3, 2, 1, 1, torch.float16,  id="c64-l512-k3-B4-fp16"),

    # -----------------------------------------------------------------------
    # AC baseline shapes
    # -----------------------------------------------------------------------
    pytest.param(1, 64, 1024, 2, 2, 0, 1, torch.float16,  id="ac-fp16-c64-l1024-k2"),
    pytest.param(1, 64, 1024, 2, 2, 0, 1, torch.bfloat16, id="ac-bf16-c64-l1024-k2"),
    pytest.param(1, 128, 512, 2, 2, 0, 1, torch.float16,  id="ac-fp16-c128-l512-k2"),
    pytest.param(1, 128, 512, 2, 2, 0, 1, torch.bfloat16, id="ac-bf16-c128-l512-k2"),
    pytest.param(1, 256, 256, 2, 2, 0, 1, torch.float16,  id="ac-fp16-c256-l256-k2"),
    pytest.param(1, 256, 256, 2, 2, 0, 1, torch.bfloat16, id="ac-bf16-c256-l256-k2"),

    # -----------------------------------------------------------------------
    # Large channel — small length (GPU saturation via channel depth)
    # -----------------------------------------------------------------------
    pytest.param(1, 512, 256, 2, 2, 0, 1, torch.float16, id="deep-C512-l256-k2"),
    pytest.param(1, 1024, 256, 2, 2, 0, 1, torch.float16, id="deep-C1024-l256-k2"),

    # -----------------------------------------------------------------------
    # Large kernel sizes (same-padding, stride=1)
    # -----------------------------------------------------------------------
    pytest.param(1, 64, 512, 3,  1, 1, 1, torch.float16, id="l512-k3"),
    pytest.param(1, 32, 512, 5,  1, 2, 1, torch.float16, id="l512-k5"),
    pytest.param(1, 32, 512, 7,  1, 3, 1, torch.float16, id="l512-k7"),
    pytest.param(1, 32, 512, 9,  1, 4, 1, torch.float16, id="l512-k9"),
    pytest.param(1, 32, 512, 11, 1, 5, 1, torch.float16, id="l512-k11"),

    # -----------------------------------------------------------------------
    # Dilation > 1
    # -----------------------------------------------------------------------
    pytest.param(1, 64, 512, 3, 1, 1, 2, torch.float16, id="l512-k3-d2"),
    pytest.param(1, 64, 512, 3, 1, 1, 4, torch.float16, id="l512-k3-d4"),
    pytest.param(1, 64, 512, 5, 1, 2, 2, torch.float16, id="l512-k5-d2"),

    # -----------------------------------------------------------------------
    # Large length
    # -----------------------------------------------------------------------
    pytest.param(1, 64, 2048, 2, 2, 0, 1, torch.float16, id="l2048-k2"),
    pytest.param(1, 64, 4096, 2, 2, 0, 1, torch.float16, id="l4096-k2"),

    # -----------------------------------------------------------------------
    # Batch > 1
    # -----------------------------------------------------------------------
    pytest.param(4, 64, 1024, 2, 2, 0, 1, torch.float16, id="B4-c64-l1024-k2"),
    pytest.param(8, 64, 1024, 2, 2, 0, 1, torch.float16, id="B8-c64-l1024-k2"),
]


@pytest.mark.parametrize(
    "batch, channels, in_length, kernel_size, stride, padding, dilation, dtype",
    _MAX_POOL1D_BENCH_PARAMS,
)
def bench_max_pool1d(
    batch: int,
    channels: int,
    in_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: torch.dtype,
) -> None:
    """Benchmark MaxPooling1dFwdOp against PyTorch."""
    out_length = _out(in_length, kernel_size, stride, padding, dilation)

    class _T:
        pass

    t = _T()
    t.batch = batch
    t.channels = channels
    t.in_length = in_length
    t.out_length = out_length
    t.kernel_size = kernel_size
    t.stride = stride
    t.padding = padding
    t.dilation = dilation

    op = MaxPooling1dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
    )
    x = torch.randn(batch, channels, in_length, dtype=dtype, device="cuda")
    bm = MaxPooling1dBenchmark(t)

    BenchmarkReport.record(op, locals(), bm.profile(op, x), tag="tileops")
    BenchmarkReport.record(
        op, locals(),
        bm.profile(lambda: torch.nn.functional.max_pool1d(
            x,
            kernel_size=kernel_size,
            stride=stride if stride else kernel_size,
            padding=padding if padding else 0,
            dilation=dilation if dilation else 1,
        )),
        tag="torch",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
