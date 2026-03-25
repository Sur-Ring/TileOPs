"""Benchmarks for MaxPooling2dFwdOp."""

from typing import Optional, Union

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MaxPooling2dFwdOp


class MaxPooling2dBenchmark(BenchmarkBase):
    """Benchmark for MaxPooling2dFwdOp."""

    def __init__(self, test):
        super().__init__(test)
        self.batch = test.batch
        self.channels = test.channels
        self.in_h = test.in_h
        self.in_w = test.in_w
        self.out_h = test.out_h
        self.out_w = test.out_w
        self.kernel_size = test.kernel_size
        self.stride = test.stride
        self.padding = test.padding
        self.dilation = test.dilation

    def calculate_flops(self) -> Optional[float]:
        """Max pooling performs (kh * kw - 1) comparisons per output element."""
        if isinstance(self.kernel_size, int):
            kh = kw = self.kernel_size
        else:
            kh, kw = self.kernel_size
        return self.batch * self.channels * self.out_h * self.out_w * (kh * kw - 1)

    def calculate_memory(self) -> Optional[float]:
        """Read input + write output (fp16/bf16 = 2 bytes each)."""
        bpe = 2
        read = self.batch * self.channels * self.in_h * self.in_w * bpe
        write = self.batch * self.channels * self.out_h * self.out_w * bpe
        return read + write


def _out(length: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


_MAX_POOL2D_BENCH_PARAMS = [
    # -----------------------------------------------------------------------
    # AC baseline shapes (issue #2 acceptance criteria)
    # -----------------------------------------------------------------------
    pytest.param(1, 64, 224, 224, 2, 2, 0, 1, torch.float16,  id="ac-fp16-224x224-k2"),
    pytest.param(1, 64, 224, 224, 2, 2, 0, 1, torch.bfloat16, id="ac-bf16-224x224-k2"),
    pytest.param(1, 128, 112, 112, 2, 2, 0, 1, torch.float16, id="ac-fp16-112x112-k2"),
    pytest.param(1, 128, 112, 112, 2, 2, 0, 1, torch.bfloat16, id="ac-bf16-112x112-k2"),
    pytest.param(1, 256, 56, 56, 2, 2, 0, 1, torch.float16,  id="ac-fp16-56x56-k2"),
    pytest.param(1, 256, 56, 56, 2, 2, 0, 1, torch.bfloat16, id="ac-bf16-56x56-k2"),

    # -----------------------------------------------------------------------
    # Large channel — small spatial (GPU saturation via channel depth)
    # -----------------------------------------------------------------------
    pytest.param(1, 512,  56,  56, 2, 2, 0, 1, torch.float16, id="deep-C512-56x56-k2"),
    pytest.param(1, 1024, 56,  56, 2, 2, 0, 1, torch.float16, id="deep-C1024-56x56-k2"),
    pytest.param(1, 512,  112, 112, 2, 2, 0, 1, torch.float16, id="deep-C512-112x112-k2"),

    # -----------------------------------------------------------------------
    # Large kernel sizes (same-padding, stride=1)
    # -----------------------------------------------------------------------
    pytest.param(1, 64, 224, 224, 3,  1, 1, 1, torch.float16, id="224x224-k3"),
    pytest.param(1, 32, 224, 224, 5,  1, 2, 1, torch.float16, id="224x224-k5"),
    pytest.param(1, 32, 224, 224, 7,  1, 3, 1, torch.float16, id="224x224-k7"),
    pytest.param(1, 32, 224, 224, 9,  1, 4, 1, torch.float16, id="224x224-k9"),
    pytest.param(1, 32, 224, 224, 11, 1, 5, 1, torch.float16, id="224x224-k11"),
    pytest.param(1, 32, 224, 224, 13, 1, 6, 1, torch.float16, id="224x224-k13"),

    # -----------------------------------------------------------------------
    # Dilation > 1 (cuDNN has no specialised path; TileOPs wins by ~4-5x)
    # -----------------------------------------------------------------------
    pytest.param(1, 64, 224, 224, 3, 1, 1, 2, torch.float16, id="224x224-k3-d2"),
    pytest.param(1, 64, 224, 224, 3, 1, 1, 4, torch.float16, id="224x224-k3-d4"),
    pytest.param(1, 64, 224, 224, 5, 1, 2, 2, torch.float16, id="224x224-k5-d2"),
    pytest.param(1, 64, 224, 224, 7, 1, 3, 2, torch.float16, id="224x224-k7-d2"),

    # -----------------------------------------------------------------------
    # Large spatial
    # -----------------------------------------------------------------------
    pytest.param(1, 64, 448,  448,  2, 2, 0, 1, torch.float16, id="448x448-k2"),
    pytest.param(1, 64, 448,  448,  3, 2, 1, 1, torch.float16, id="448x448-k3"),
    pytest.param(1, 64, 1024, 1024, 2, 2, 0, 1, torch.float16, id="1024x1024-k2"),
    pytest.param(1, 32, 1024, 1024, 7, 1, 3, 1, torch.float16, id="1024x1024-k7"),

    # -----------------------------------------------------------------------
    # Batch > 1
    # -----------------------------------------------------------------------
    pytest.param(4, 64, 224, 224, 2, 2, 0, 1, torch.float16, id="B4-224x224-k2"),
    pytest.param(8, 64, 224, 224, 2, 2, 0, 1, torch.float16, id="B8-224x224-k2"),

    # -----------------------------------------------------------------------
    # Rectangular kernel / misc
    # -----------------------------------------------------------------------
    pytest.param(1, 3, 224, 224, (2, 3), (2, 2), (0, 1), 1, torch.float16, id="rectangular"),
    pytest.param(1, 16, 64, 64, 4, 4, 0, 1, torch.float16, id="stride4"),
]


@pytest.mark.parametrize(
    "batch, channels, in_h, in_w, kernel_size, stride, padding, dilation, dtype",
    _MAX_POOL2D_BENCH_PARAMS,
)
def bench_max_pool2d(
    batch: int,
    channels: int,
    in_h: int,
    in_w: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    dilation: Union[int, tuple[int, int]],
    dtype: torch.dtype,
) -> None:
    """Benchmark MaxPooling2dFwdOp against PyTorch."""
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size
    if isinstance(stride, int):
        sh = sw = stride
    else:
        sh, sw = stride
    if isinstance(padding, int):
        ph = pw = padding
    else:
        ph, pw = padding
    if isinstance(dilation, int):
        dh = dw = dilation
    else:
        dh, dw = dilation

    out_h = _out(in_h, kh, sh, ph, dh)
    out_w = _out(in_w, kw, sw, pw, dw)

    class _T:
        pass

    t = _T()
    t.batch = batch
    t.channels = channels
    t.in_h = in_h
    t.in_w = in_w
    t.out_h = out_h
    t.out_w = out_w
    t.kernel_size = kernel_size
    t.stride = stride
    t.padding = padding
    t.dilation = dilation

    op = MaxPooling2dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
    )
    x = torch.randn(batch, channels, in_h, in_w, dtype=dtype, device="cuda")
    bm = MaxPooling2dBenchmark(t)

    BenchmarkReport.record(op, locals(), bm.profile(op, x), tag="tileops")
    BenchmarkReport.record(
        op, locals(),
        bm.profile(lambda: torch.nn.functional.max_pool2d(
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
