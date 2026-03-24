"""Tests for Average Pooling 1D Forward Op."""

from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import AvgPooling1dOp


class AvgPooling1dFixture(FixtureBase):
    PARAMS = [
        (
            "batch, channels, in_seq_len, kernel_size, stride, padding, dilation, dtype, tune",
            [
                # Smoke tests - MUST come first
                pytest.param(
                    1, 3, 100, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1, 3, 100, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.smoke,
                ),
                # Full tests
                pytest.param(
                    1, 3, 100, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 16, 56, 3, 1, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 32, 2, 2, 0, 1, torch.float16, True,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 3, 100, 3, 2, 1, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 16, 56, 3, 1, 1, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),
                # Different stride tests
                pytest.param(
                    1, 8, 64, 4, 4, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                # Dilation tests
                pytest.param(
                    1, 4, 32, 3, 1, 2, 2, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                # Padding tests
                pytest.param(
                    1, 4, 50, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


class AvgPooling1dTest(TestBase):
    """Test class for AvgPooling1dOp."""

    def __init__(
        self,
        batch: int,
        channels: int,
        in_seq_len: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        dtype: torch.dtype,
        tune: bool,
    ):
        self.batch = batch
        self.channels = channels
        self.in_seq_len = in_seq_len
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dtype = dtype
        self.tune = tune

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        """Generate random input tensor."""
        x = torch.randn(
            self.batch, self.channels, self.in_seq_len,
            device='cuda', dtype=self.dtype,
        )
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        """Reference implementation using torch.nn.functional.avg_pool1d."""
        y = torch.nn.functional.avg_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        return y


@AvgPooling1dFixture
def test_avg_pool1d_fwd(
    batch: int,
    channels: int,
    in_seq_len: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Test AvgPooling1dOp correctness."""
    test = AvgPooling1dTest(
        batch=batch,
        channels=channels,
        in_seq_len=in_seq_len,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        tune=tune,
    )

    op = AvgPooling1dOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        tune=tune,
    )

    inputs = test.gen_inputs()

    # Use appropriate tolerances based on dtype
    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:  # bfloat16
        atol, rtol = 1.6e-2, 1.6e-2

    test.check(op, *inputs, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
