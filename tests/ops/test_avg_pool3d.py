"""Tests for Average Pooling 3D Forward Op."""

from typing import Tuple, Union

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import AvgPooling3dOp


class AvgPooling3dFixture(FixtureBase):
    PARAMS = [
        (
            "batch, channels, in_d, in_h, in_w, kernel_size, stride, padding, dilation, dtype, tune",
            [
                # Smoke tests - MUST come first
                pytest.param(
                    1, 3, 8, 32, 32, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1, 3, 8, 32, 32, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.smoke,
                ),
                # Full tests
                pytest.param(
                    1, 3, 8, 32, 32, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 8, 4, 16, 16, 2, 1, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 3, 8, 32, 32, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),
                # Rectangular kernel tests
                pytest.param(
                    1, 3, 4, 16, 16, (2, 2, 2), 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                # Different stride tests
                pytest.param(
                    1, 4, 8, 32, 32, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


class AvgPooling3dTest(TestBase):
    """Test class for AvgPooling3dOp."""

    def __init__(
        self,
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
        tune: bool,
    ):
        self.batch = batch
        self.channels = channels
        self.in_d = in_d
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
            self.batch, self.channels, self.in_d, self.in_h, self.in_w,
            device='cuda', dtype=self.dtype,
        )
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        """Reference implementation using torch.nn.functional.avg_pool3d."""
        # NOTE: PyTorch's avg_pool3d does NOT support dilation.
        y = torch.nn.functional.avg_pool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        return y


@AvgPooling3dFixture
def test_avg_pool3d_fwd(
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
    tune: bool,
) -> None:
    """Test AvgPooling3dOp correctness."""
    test = AvgPooling3dTest(
        batch=batch,
        channels=channels,
        in_d=in_d,
        in_h=in_h,
        in_w=in_w,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=dtype,
        tune=tune,
    )

    op = AvgPooling3dOp(
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
