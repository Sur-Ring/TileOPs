"""Tests for Max Pooling 1D Forward Op."""

from typing import Tuple, Union

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MaxPooling1dFwdOp


class MaxPooling1dFixture(FixtureBase):
    PARAMS = [
        (
            "batch, channels, in_length, kernel_size, stride, padding, dilation, dtype, tune",
            [
                # ------------------------------------------------------------------
                # Smoke tests — minimal set, must come first
                # ------------------------------------------------------------------
                pytest.param(
                    1, 3, 224, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1, 3, 224, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.smoke,
                ),

                # ------------------------------------------------------------------
                # Full tests
                # ------------------------------------------------------------------

                # kernel=3, s=2, p=1
                pytest.param(
                    1, 64, 512, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 512, 3, 2, 1, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),

                # AC baseline shapes — fp16 and bf16
                pytest.param(
                    1, 64, 1024, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 1024, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 128, 512, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 128, 512, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 256, 256, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 256, 256, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),

                # Different kernel sizes with same-padding (stride=1)
                pytest.param(
                    1, 32, 512, 3, 1, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 32, 512, 5, 1, 2, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 32, 512, 7, 1, 3, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Padding variation
                pytest.param(
                    1, 64, 512, 3, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 512, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Dilation > 1
                pytest.param(
                    1, 4, 64, 3, 1, 1, 2, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 512, 3, 1, 1, 2, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 512, 5, 1, 2, 2, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Large channel
                pytest.param(
                    1, 512, 256, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Large kernel
                pytest.param(
                    1, 64, 512, 9, 1, 4, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Different stride
                pytest.param(
                    1, 8, 256, 4, 4, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Batch > 1
                pytest.param(
                    4, 32, 512, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 16, 256, 3, 1, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Autotune
                pytest.param(
                    1, 64, 128, 2, 2, 0, 1, torch.float16, True,
                    marks=pytest.mark.full,
                ),

                # ------------------------------------------------------------------
                # Nightly tests — very large lengths, slow to compile + run
                # ------------------------------------------------------------------
                pytest.param(
                    1, 64, 4096, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.nightly,
                ),
                pytest.param(
                    1, 32, 8192, 7, 1, 3, 1, torch.float16, False,
                    marks=pytest.mark.nightly,
                ),
            ],
        ),
    ]


class MaxPooling1dTest(TestBase):
    """Test class for MaxPooling1dFwdOp."""

    def __init__(
        self,
        batch: int,
        channels: int,
        in_length: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        dtype: torch.dtype,
        tune: bool,
    ):
        self.batch = batch
        self.channels = channels
        self.in_length = in_length
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dtype = dtype
        self.tune = tune

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        """Generate random input tensor."""
        x = torch.randn(
            self.batch, self.channels, self.in_length,
            device='cuda', dtype=self.dtype,
        )
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        """Reference implementation using torch.nn.functional.max_pool1d."""
        return torch.nn.functional.max_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


@MaxPooling1dFixture
def test_max_pool1d_fwd(
    batch: int,
    channels: int,
    in_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Test MaxPooling1dFwdOp correctness."""
    test = MaxPooling1dTest(
        batch=batch,
        channels=channels,
        in_length=in_length,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        tune=tune,
    )

    op = MaxPooling1dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        tune=tune,
    )

    inputs = test.gen_inputs()

    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:  # bfloat16
        atol, rtol = 1.6e-2, 1.6e-2

    test.check(op, *inputs, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
