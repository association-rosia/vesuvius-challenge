from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import RandomPerspective as RP
import torch

import numpy as np

from ttach.base import DualTransform


from typing import List

# TODO
# T.ElasticTransform(alpha=500.0, sigma=10.0),
# T.RandomHorizontalFlip(),
# T.RandomVerticalFlip()


class TTARandomRotation(DualTransform):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degree int: The range of degree (-degree, +degree).

    """

    identity_param = 0

    def __init__(
        self,
        degree: int,
    ):
        self.degree = degree
        self.angle = float(
            torch.empty(1).uniform_(float(-degree), float(degree)).item()
        )

        angles = (
            [self.angle]
            if self.identity_param == self.angle
            else [self.identity_param, self.angle]
        )

        super().__init__("angle", angles)

    def apply_aug_image(
        self,
        image,
        angle=0,
        interpolation=F.InterpolationMode.NEAREST,
        expand=False,
        center=None,
        fill=0,
        **kwargs,
    ):
        return F.rotate(image, angle, interpolation, expand, center, fill)

    def apply_deaug_mask(self, mask, angle=0, **kwargs):
        return self.apply_aug_image(mask, -angle)

    def apply_deaug_label(self, label, angle=0, **kwargs):
        return label


class TTARandomPerspective(DualTransform):
    """Performs a random perspective transformation of the given image with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
            Default is 0.5.

    """

    identity_param = 0

    def __init__(self, distortion_scale=0.5):
        self.distortion_scale = distortion_scale

        distortion_scales = (
            [self.distortion_scale]
            if self.identity_param == self.distortion_scale
            else [self.identity_param, self.distortion_scale]
        )

        self.topleft_factors = [
            float(torch.empty(1).uniform_(0.0, 1.0).item()),
            float(torch.empty(1).uniform_(0.0, 1.0).item()),
        ]
        self.topright_factors = [
            float(torch.empty(1).uniform_(0.0, 1.0).item()),
            float(torch.empty(1).uniform_(0.0, 1.0).item()),
        ]
        self.botright_factors = [
            float(torch.empty(1).uniform_(0.0, 1.0).item()),
            float(torch.empty(1).uniform_(0.0, 1.0).item()),
        ]
        self.botleft_factors = [
            float(torch.empty(1).uniform_(0.0, 1.0).item()),
            float(torch.empty(1).uniform_(0.0, 1.0).item()),
        ]

        super().__init__("distortion_scale", distortion_scales)

    def _get_points(self, image):
        _, height, width = F.get_dimensions(image)

        half_height = height // 2
        half_width = width // 2

        startpoints = [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]

        endpoints = [
            np.multiply(
                self.topleft_factors,
                [
                    half_width * self.distortion_scale,
                    half_height * self.distortion_scale,
                ],
            )
            .astype("int")
            .tolist(),
            np.add(
                [width, 0],
                np.multiply(
                    self.topright_factors,
                    [
                        -half_width * self.distortion_scale,
                        half_height * self.distortion_scale,
                    ],
                ),
            )
            .astype("int")
            .tolist(),
            np.add(
                [width, height],
                np.multiply(
                    self.botright_factors,
                    [
                        -half_width * self.distortion_scale,
                        -half_height * self.distortion_scale,
                    ],
                ),
            )
            .astype("int")
            .tolist(),
            np.add(
                [0, height],
                np.multiply(
                    self.botleft_factors,
                    [
                        half_width * self.distortion_scale,
                        -half_height * self.distortion_scale,
                    ],
                ),
            )
            .astype("int")
            .tolist(),
        ]

        return startpoints, endpoints

    def apply_aug_image(
        self,
        image,
        distortion_scale,
        interpolation=F.InterpolationMode.BILINEAR,
        fill=0,
        **kwargs,
    ):
        startpoints, endpoints = self._get_points(image)

        return F.perspective(image, startpoints, endpoints, interpolation, fill)

    def apply_deaug_mask(
        self,
        image,
        distortion_scale,
        interpolation=F.InterpolationMode.BILINEAR,
        fill=0,
        **kwargs,
    ):
        startpoints, endpoints = self._get_points(image)

        return F.perspective(image, endpoints, startpoints, interpolation, fill)

    def apply_deaug_label(self, label, **kwargs):
        return label
