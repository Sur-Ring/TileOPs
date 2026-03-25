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
                # ------------------------------------------------------------------
                # Smoke tests — minimal set, must come first
                # ------------------------------------------------------------------
                pytest.param(
                    1, 3, 224, 224, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1, 3, 224, 224, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.smoke,
                ),

                # ------------------------------------------------------------------
                # Full tests
                # ------------------------------------------------------------------

                # ResNet stem max-pool: k=3, s=2, p=1
                pytest.param(
                    1, 64, 112, 112, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 112, 112, 3, 2, 1, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),

                # AC baseline shapes (224/112/56) — fp16 and bf16
                pytest.param(
                    1, 64, 224, 224, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 224, 224, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 128, 112, 112, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 128, 112, 112, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 256, 56, 56, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 256, 56, 56, 2, 2, 0, 1, torch.bfloat16, False,
                    marks=pytest.mark.full,
                ),

                # Large channel — small spatial saturated by channel depth
                pytest.param(
                    1, 512, 56, 56, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 256, 112, 112, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 512, 112, 112, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Different kernel sizes with same-padding (stride=1)
                pytest.param(
                    1, 32, 224, 224, 3, 1, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 32, 224, 224, 5, 1, 2, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 32, 224, 224, 7, 1, 3, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 32, 224, 224, 9, 1, 4, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 32, 224, 224, 11, 1, 5, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 32, 224, 224, 13, 1, 6, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Small kernel, large channel — k=3
                pytest.param(
                    1, 256, 56, 56, 3, 1, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 512, 56, 56, 3, 1, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Padding variation
                pytest.param(
                    1, 64, 224, 224, 3, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 224, 224, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 224, 224, 7, 2, 3, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Dilation (padding must be <= kernel_size // 2 for PyTorch)
                pytest.param(
                    1, 4, 32, 32, 3, 1, 1, 2, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 224, 224, 3, 1, 1, 2, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 224, 224, 3, 1, 1, 4, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 224, 224, 5, 1, 2, 2, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Large spatial
                pytest.param(
                    1, 64, 448, 448, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 64, 448, 448, 3, 2, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Rectangular kernel
                pytest.param(
                    1, 3, 112, 112, (2, 3), (2, 2), (0, 1), 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Different stride
                pytest.param(
                    1, 8, 64, 64, 4, 4, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Batch > 1
                pytest.param(
                    4, 32, 112, 112, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 16, 56, 56, 3, 1, 1, 1, torch.float16, False,
                    marks=pytest.mark.full,
                ),

                # Autotune
                pytest.param(
                    1, 64, 32, 32, 2, 2, 0, 1, torch.float16, True,
                    marks=pytest.mark.full,
                ),

                # ------------------------------------------------------------------
                # Nightly tests — very large spatial, slow to compile + run
                # ------------------------------------------------------------------
                pytest.param(
                    1, 64, 1024, 1024, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.nightly,
                ),
                pytest.param(
                    1, 32, 1024, 1024, 7, 1, 3, 1, torch.float16, False,
                    marks=pytest.mark.nightly,
                ),
                pytest.param(
                    1, 32, 2048, 2048, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.nightly,
                ),
                pytest.param(
                    1, 16, 4096, 4096, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.nightly,
                ),
                pytest.param(
                    1, 8, 8192, 8192, 2, 2, 0, 1, torch.float16, False,
                    marks=pytest.mark.nightly,
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
        return torch.nn.functional.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


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

    if dtype == torch.float16:
        atol, rtol = 1e-3, 1e-3
    else:  # bfloat16
        atol, rtol = 1.6e-2, 1.6e-2

    test.check(op, *inputs, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
