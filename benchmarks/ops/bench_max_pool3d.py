"""Benchmarks for MaxPooling3dFwdOp."""

from typing import Optional, Union

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MaxPooling3dFwdOp


class MaxPooling3dBenchmark(BenchmarkBase):
    """Benchmark for MaxPooling3dFwdOp."""

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
        self.stride = test.stride
        self.padding = test.padding
        self.dilation = test.dilation

    def calculate_flops(self) -> Optional[float]:
        """Max pooling performs (kd * kh * kw - 1) comparisons per output element."""
        if isinstance(self.kernel_size, int):
            kd = kh = kw = self.kernel_size
        else:
            kd, kh, kw = self.kernel_size
        return self.batch * self.channels * self.out_d * self.out_h * self.out_w * (kd * kh * kw - 1)

    def calculate_memory(self) -> Optional[float]:
        """Read input + write output (fp16/bf16 = 2 bytes each)."""
        bpe = 2
        read = self.batch * self.channels * self.in_d * self.in_h * self.in_w * bpe
        write = self.batch * self.channels * self.out_d * self.out_h * self.out_w * bpe
        return read + write


def _out(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


_MAX_POOL3D_BENCH_PARAMS = [
    # -----------------------------------------------------------------------
    # Common 3D pooling shapes (video/volumetric CNNs)
    # -----------------------------------------------------------------------
    pytest.param(1, 16, 16, 56, 56, 3, 2, 1, 1, torch.float16,  id="c16-d16-56x56-k3-fp16"),
    pytest.param(1, 16, 16, 56, 56, 3, 2, 1, 1, torch.bfloat16, id="c16-d16-56x56-k3-bf16"),
    pytest.param(4, 16, 16, 56, 56, 3, 2, 1, 1, torch.float16,  id="c16-d16-56x56-k3-B4-fp16"),

    # -----------------------------------------------------------------------
    # AC baseline shapes
    # -----------------------------------------------------------------------
    pytest.param(1, 32, 16, 32, 32, 2, 2, 0, 1, torch.float16,  id="ac-fp16-c32-d16-32x32-k2"),
    pytest.param(1, 32, 16, 32, 32, 2, 2, 0, 1, torch.bfloat16, id="ac-bf16-c32-d16-32x32-k2"),
    pytest.param(1, 64, 8, 28, 28, 2, 2, 0, 1, torch.float16,   id="ac-fp16-c64-d8-28x28-k2"),
    pytest.param(1, 64, 8, 28, 28, 2, 2, 0, 1, torch.bfloat16,  id="ac-bf16-c64-d8-28x28-k2"),

    # -----------------------------------------------------------------------
    # Large channel — small spatial
    # -----------------------------------------------------------------------
    pytest.param(1, 128, 8, 16, 16, 2, 2, 0, 1, torch.float16, id="deep-C128-d8-16x16-k2"),
    pytest.param(1, 256, 4, 16, 16, 2, 2, 0, 1, torch.float16, id="deep-C256-d4-16x16-k2"),

    # -----------------------------------------------------------------------
    # Large kernel sizes (same-padding, stride=1)
    # -----------------------------------------------------------------------
    pytest.param(1, 8, 8, 32, 32, 3, 1, 1, 1, torch.float16, id="32x32-k3"),
    pytest.param(1, 8, 8, 32, 32, 5, 1, 2, 1, torch.float16, id="32x32-k5"),

    # -----------------------------------------------------------------------
    # Dilation > 1
    # -----------------------------------------------------------------------
    pytest.param(1, 8, 8, 32, 32, 3, 1, 1, 2, torch.float16, id="32x32-k3-d2"),

    # -----------------------------------------------------------------------
    # Large spatial, stride=1 (TileOPs advantage at high output counts)
    # -----------------------------------------------------------------------
    pytest.param(1, 16,  8,  56,  56,  3, 1, 1, 1, torch.float16, id="c16-d8-56x56-k3"),
    pytest.param(1, 32,  8,  56,  56,  3, 1, 1, 1, torch.float16, id="c32-d8-56x56-k3"),
    pytest.param(1, 32,  8, 112, 112,  3, 1, 1, 1, torch.float16, id="c32-d8-112x112-k3"),
    pytest.param(1, 16, 16, 112, 112,  3, 1, 1, 1, torch.float16, id="c16-d16-112x112-k3"),
    pytest.param(1,  8,  8, 112, 112,  3, 1, 1, 1, torch.float16, id="c8-d8-112x112-k3"),
    pytest.param(1,  8,  8, 224, 224,  3, 1, 1, 1, torch.float16, id="c8-d8-224x224-k3"),
    pytest.param(1, 16, 32,  56,  56,  3, 1, 1, 1, torch.float16, id="c16-d32-56x56-k3"),
    pytest.param(1, 16, 64,  32,  32,  3, 1, 1, 1, torch.float16, id="c16-d64-32x32-k3"),
    pytest.param(1, 16,  8,  56,  56,  5, 1, 2, 1, torch.float16, id="c16-d8-56x56-k5"),
    pytest.param(1,  8,  8,  56,  56,  7, 1, 3, 1, torch.float16, id="c8-d8-56x56-k7"),
    pytest.param(1,  8,  8,  56,  56,  3, 1, 1, 2, torch.float16, id="c8-d8-56x56-k3-d2"),
    pytest.param(1,  8, 32, 112, 112,  3, 2, 1, 1, torch.float16, id="c8-d32-112x112-k3-s2"),

    # -----------------------------------------------------------------------
    # Batch > 1
    # -----------------------------------------------------------------------
    pytest.param(4, 8, 8, 16, 16, 2, 2, 0, 1, torch.float16, id="B4-c8-d8-16x16-k2"),
    pytest.param(8, 8, 8, 16, 16, 2, 2, 0, 1, torch.float16, id="B8-c8-d8-16x16-k2"),
    pytest.param(4, 8, 8, 56, 56, 3, 1, 1, 1, torch.float16, id="B4-c8-d8-56x56-k3"),
    pytest.param(8, 8, 8, 56, 56, 3, 1, 1, 1, torch.float16, id="B8-c8-d8-56x56-k3"),

    # -----------------------------------------------------------------------
    # Non-cubic kernel
    # -----------------------------------------------------------------------
    pytest.param(1, 8, 16, 32, 32, (2, 3, 3), (2, 2, 2), (0, 1, 1), 1, torch.float16, id="non-cubic"),
]


@pytest.mark.parametrize(
    "batch, channels, in_d, in_h, in_w, kernel_size, stride, padding, dilation, dtype",
    _MAX_POOL3D_BENCH_PARAMS,
)
def bench_max_pool3d(
    batch: int,
    channels: int,
    in_d: int,
    in_h: int,
    in_w: int,
    kernel_size: Union[int, tuple],
    stride: Union[int, tuple],
    padding: Union[int, tuple],
    dilation: Union[int, tuple],
    dtype: torch.dtype,
) -> None:
    """Benchmark MaxPooling3dFwdOp against PyTorch."""
    if isinstance(kernel_size, int):
        kd = kh = kw = kernel_size
    else:
        kd, kh, kw = kernel_size
    if isinstance(stride, int):
        sd = sh = sw = stride
    else:
        sd, sh, sw = stride
    if isinstance(padding, int):
        pd = ph = pw = padding
    else:
        pd, ph, pw = padding
    if isinstance(dilation, int):
        dd = dh = dw = dilation
    else:
        dd, dh, dw = dilation

    out_d = _out(in_d, kd, sd, pd, dd)
    out_h = _out(in_h, kh, sh, ph, dh)
    out_w = _out(in_w, kw, sw, pw, dw)

    class _T:
        pass

    t = _T()
    t.batch = batch
    t.channels = channels
    t.in_d = in_d
    t.in_h = in_h
    t.in_w = in_w
    t.out_d = out_d
    t.out_h = out_h
    t.out_w = out_w
    t.kernel_size = kernel_size
    t.stride = stride
    t.padding = padding
    t.dilation = dilation

    op = MaxPooling3dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
    )
    x = torch.randn(batch, channels, in_d, in_h, in_w, dtype=dtype, device="cuda")
    bm = MaxPooling3dBenchmark(t)

    BenchmarkReport.record(op, locals(), bm.profile(op, x), tag="tileops")
    BenchmarkReport.record(
        op, locals(),
        bm.profile(lambda: torch.nn.functional.max_pool3d(
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
