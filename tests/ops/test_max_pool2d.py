"""Tests for Max Pooling 2D Forward Op."""

from typing import Tuple, Union

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MaxPooling2dFwdOp


class MaxPooling2dFixture(FixtureBase):
    PARAMS = [
        (
            "batch, channels, in_h, in_w, kernel_size, stride, padding, dilation, dtype, tune",
            [
                # Smoke tests - MUST come first
                pytest.param(
                    1, 3, 224, 224, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1, 3, 224, 224, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.smoke,
                ),
                # Full tests
                pytest.param(
                    1, 3, 224, 224, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 16, 56, 56, 3, 1, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 32, 32, 2, 2, 0, 1, torch.float16, True,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 3, 224, 224, 3, 2, 1, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 16, 56, 56, 3, 1, 1, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),
                # Rectangular kernel tests
                pytest.param(
                    1, 3, 112, 112, (2, 3), (2, 2), (0, 1), 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                # Different stride tests
                pytest.param(
                    1, 8, 64, 64, 4, 4, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                # Dilation tests
                pytest.param(
                    1, 4, 32, 32, 3, 1, 2, 2, torch.float16, False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


class MaxPooling2dTest(TestBase):
    """Test class for MaxPooling2dFwdOp."""

    def __init__(
        self,
        batch: int,
        channels: int,
        in_h: int,
        in_w: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]],
        padding: Union[int, tuple[int, int]],
        dilation: Union[int, tuple[int, int]],
        dtype: torch.dtype,
        tune: bool,
    ):
        self.batch = batch
        self.channels = channels
        self.in_h = in_h
        self.in_w = in_w
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dtype = dtype
        self.tune = tune

    def gen_inputs(self) -> Tuple[torch.Tensor]:
        """Generate random input tensor."""
        x = torch.randn(
            self.batch, self.channels, self.in_h, self.in_w,
            device='cuda', dtype=self.dtype,
        )
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        """Reference implementation using torch.nn.functional.max_pool2d."""
        # Handle kernel_size as tuple or int
        if isinstance(self.kernel_size, int):
            kernel_size = self.kernel_size
        else:
            kernel_size = self.kernel_size

        # Handle stride
        if self.stride is None:
            stride = self.kernel_size
        elif isinstance(self.stride, int):
            stride = self.stride
        else:
            stride = self.stride

        # Handle padding
        if self.padding is None:
            padding = 0
        elif isinstance(self.padding, int):
            padding = self.padding
        else:
            padding = self.padding

        # Handle dilation
        if self.dilation is None:
            dilation = 1
        elif isinstance(self.dilation, int):
            dilation = self.dilation
        else:
            dilation = self.dilation

        y = torch.nn.functional.max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        return y


@MaxPooling2dFixture
def test_max_pool2d_fwd(
    batch: int,
    channels: int,
    in_h: int,
    in_w: int,
    kernel_size: Union[int, tuple[int, int]],
    stride: Union[int, tuple[int, int]],
    padding: Union[int, tuple[int, int]],
    dilation: Union[int, tuple[int, int]],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Test MaxPooling2dFwdOp correctness."""
    test = MaxPooling2dTest(
        batch=batch,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        tune=tune,
    )

    op = MaxPooling2dFwdOp(
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