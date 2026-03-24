"""Tests for MaxPoolingFwdOp."""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.pooling import MaxPoolingFwdOp


class MaxPoolingFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, in_seq_len, dim, kernel_size, stride, padding, dilation, dtype, tune",
            [
                # Standard cases
                pytest.param(1, 512, 4096, 2, 2, 0, 1, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1, 512, 4096, 2, 2, 0, 1, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 512, 4096, 2, 2, 0, 1, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2, 512, 4096, 2, 2, 0, 1, torch.bfloat16, False, marks=pytest.mark.full),
                # Different kernel sizes
                pytest.param(1, 512, 4096, 4, 4, 0, 1, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1, 512, 4096, 8, 8, 0, 1, torch.float16, False, marks=pytest.mark.full),
                # With padding
                pytest.param(1, 512, 4096, 2, 2, 1, 1, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1, 512, 4096, 3, 1, 1, 1, torch.float16, False, marks=pytest.mark.full),
                # Non-power-of-two shapes
                pytest.param(1, 513, 4096, 2, 2, 0, 1, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1, 512, 3000, 2, 2, 0, 1, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1, 513, 3000, 2, 2, 0, 1, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


class MaxPoolingFwdTest(TestBase):
    def __init__(
        self,
        batch: int,
        in_seq_len: int,
        dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.in_seq_len = in_seq_len
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.batch, self.in_seq_len, self.dim,
            dtype=self.dtype, device="cuda",
        )
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch max_pool1d expects (N, C, L) or (N, L, C) depending on layout
        # Our op uses (batch, seq_len, dim) = (N, L, C)
        x_2d = x  # (batch, seq_len, dim)
        # Apply max pooling over seq_len dimension
        out = F.max_pool1d(
            x_2d.transpose(1, 2),  # (batch, dim, seq_len)
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        # Return as (batch, seq_len, dim)
        return out.transpose(1, 2).to(self.dtype)


@MaxPoolingFwdFixture
def test_max_pooling_fwd(
    batch: int,
    in_seq_len: int,
    dim: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPoolingFwdTest(
        batch, in_seq_len, dim, kernel_size, stride, padding, dilation, dtype
    )
    op = MaxPoolingFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        tune=tune,
    )
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    rtol = atol
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class MaxPoolingFwdStrideNoneFixture(FixtureBase):
    PARAMS = [
        (
            "batch, in_seq_len, dim, kernel_size, dtype",
            [
                pytest.param(1, 512, 4096, 2, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1, 512, 4096, 4, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@MaxPoolingFwdStrideNoneFixture
def test_max_pooling_fwd_stride_none(
    batch: int,
    in_seq_len: int,
    dim: int,
    kernel_size: int,
    dtype: torch.dtype,
) -> None:
    """Test that stride defaults to kernel_size when not specified."""
    test = MaxPoolingFwdTest(
        batch, in_seq_len, dim, kernel_size, kernel_size, 0, 1, dtype
    )
    op = MaxPoolingFwdOp(
        kernel_size=kernel_size,
        dtype=dtype,
    )
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    test.check(op, *test.gen_inputs(), atol=atol, rtol=atol)


class MaxPoolingFwdDilationFixture(FixtureBase):
    PARAMS = [
        (
            "batch, in_seq_len, dim, kernel_size, stride, padding, dilation, dtype",
            [
                pytest.param(1, 64, 4096, 3, 1, 0, 2, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1, 64, 4096, 3, 1, 1, 2, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@MaxPoolingFwdDilationFixture
def test_max_pooling_fwd_dilation(
    batch: int,
    in_seq_len: int,
    dim: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: torch.dtype,
) -> None:
    """Test max pooling with dilation > 1."""
    test = MaxPoolingFwdTest(
        batch, in_seq_len, dim, kernel_size, stride, padding, dilation, dtype
    )
    op = MaxPoolingFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
    )
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    test.check(op, *test.gen_inputs(), atol=atol, rtol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
