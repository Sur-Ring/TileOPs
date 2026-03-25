"""Tests for Max Pooling 3D Forward Op."""

from typing import Tuple, Union

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MaxPooling3dFwdOp


class MaxPooling3dFixture(FixtureBase):
    PARAMS = [
        (
            "batch, channels, in_d, in_h, in_w, kernel_size, stride, padding, dilation, dtype, tune",
            [
                # ------------------------------------------------------------------
                # Smoke tests — minimal set, must come first
                # ------------------------------------------------------------------
                pytest.param(
                    1, 3, 16, 16, 16, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1, 3, 16, 16, 16, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.smoke,
                ),

                # ------------------------------------------------------------------
                # Full tests
                # ------------------------------------------------------------------

                # kernel=3, s=2, p=1
                pytest.param(
                    1, 16, 16, 56, 56, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 16, 16, 56, 56, 3, 2, 1, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),

                # AC baseline shapes — fp16 and bf16
                pytest.param(
                    1, 32, 16, 32, 32, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 32, 16, 32, 32, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 8, 28, 28, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 8, 28, 28, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),

                # Different kernel sizes with same-padding (stride=1)
                pytest.param(
                    1, 8, 8, 32, 32, 3, 1, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 8, 8, 32, 32, 5, 1, 2, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Padding variation
                pytest.param(
                    1, 16, 8, 32, 32, 3, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 16, 8, 32, 32, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Dilation > 1
                pytest.param(
                    1, 4, 8, 16, 16, 3, 1, 1, 2, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Large channel
                pytest.param(
                    1, 128, 8, 16, 16, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Non-cubic kernel
                pytest.param(
                    1, 8, 16, 32, 32, (2, 3, 3), (2, 2, 2), (0, 1, 1), 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Different stride
                pytest.param(
                    1, 8, 16, 32, 32, 4, 4, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Batch > 1
                pytest.param(
                    4, 8, 8, 16, 16, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 4, 8, 16, 16, 3, 1, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Autotune
                pytest.param(
                    1, 8, 8, 16, 16, 2, 2, 0, 1, torch.float16, True,
                    marks=pytest.mark.full,
                ),

                # ------------------------------------------------------------------
                # Nightly tests — very large spatial, slow to compile + run
                # ------------------------------------------------------------------
                pytest.param(
                    1, 16, 32, 64, 64, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.nightly,
                ),
                pytest.param(
                    1, 8, 16, 128, 128, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.nightly,
                ),
            ],
        ),
    ]


class MaxPooling3dTest(TestBase):
    """Test class for MaxPooling3dFwdOp."""

    def __init__(
        self,
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
        """Reference implementation using torch.nn.functional.max_pool3d."""
        return torch.nn.functional.max_pool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


@MaxPooling3dFixture
def test_max_pool3d_fwd(
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
    tune: bool,
) -> None:
    """Test MaxPooling3dFwdOp correctness."""
    test = MaxPooling3dTest(
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

    op = MaxPooling3dFwdOp(
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
