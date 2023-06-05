from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import RandomPerspective as RP
import torch

import numpy as np

from ttach.base import DualTransform


from typing import List


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
        return self.apply_aug_image(label, -angle)


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

    def apply_deaug_label(
        self,
        image,
        distortion_scale,
        interpolation=F.InterpolationMode.BILINEAR,
        fill=0,
        **kwargs,
    ):
        startpoints, endpoints = self._get_points(image)

        return F.perspective(image, endpoints, startpoints, interpolation, fill)


class TTAElasticTransform(DualTransform):
    identity_param = [0.0, 0.0]

    def __init__(self, alpha_sigma=[50.0, 5.0]):
        self.alpha_sigma = alpha_sigma

        alphas_sigmas = (
            [self.alpha_sigma]
            if self.identity_param == self.alpha_sigma
            else [self.identity_param, self.alpha_sigma]
        )

        super().__init__("alpha_sigma", alphas_sigmas)

    def get_params(self, alpha: float, sigma: float, size: List[int]):
        dx = torch.rand([1, 1] + size) * 2 - 1
        if sigma > 0.0:
            kx = int(8 * sigma + 1)
            # if kernel size is even we have to make it odd
            if kx % 2 == 0:
                kx += 1
            dx = F.gaussian_blur(dx, [kx, kx], sigma)
        dx = dx * alpha / size[0]

        dy = torch.rand([1, 1] + size) * 2 - 1
        if sigma > 0.0:
            ky = int(8 * sigma + 1)
            # if kernel size is even we have to make it odd
            if ky % 2 == 0:
                ky += 1
            dy = F.gaussian_blur(dy, [ky, ky], sigma)
        dy = dy * alpha / size[1]
        return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2

    def apply_aug_image(
        self,
        image,
        alpha_sigma,
        interpolation=F.InterpolationMode.BILINEAR,
        fill=0,
        **kwargs,
    ):
        _, height, width = F.get_dimensions(image)
        displacement = self.get_params(alpha_sigma[0], alpha_sigma[1], [height, width])
        self.displacement = displacement
        return F.elastic_transform(image, displacement, interpolation, fill)

    def apply_deaug_mask(
        self,
        mask,
        interpolation=F.InterpolationMode.BILINEAR,
        fill=0,
        **kwargs,
    ):
        return F.elastic_transform(
            mask,
            -self.displacement,
            interpolation,
            fill,
        )

    def apply_deaug_label(
        self,
        label,
        interpolation=F.InterpolationMode.BILINEAR,
        fill=0,
        **kwargs,
    ):
        return F.elastic_transform(
            label,
            -self.displacement,
            interpolation,
            fill,
        )


class TTAHorizontalFlip(DualTransform):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = F.hflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = F.hflip(mask)
        return mask

    def apply_deaug_label(self, label, apply=False, **kwargs):
        if apply:
            label = F.hflip(label)
        return label

    def apply_deaug_keypoints(self, keypoints, apply=False, **kwargs):
        if apply:
            keypoints = F.keypoints_hflip(keypoints)
        return keypoints


class TTAVerticalFlip(DualTransform):
    """Flip images vertically (up->down)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = F.vflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = F.vflip(mask)
        return mask

    def apply_deaug_label(self, label, apply=False, **kwargs):
        if apply:
            label = F.vflip(label)
        return label

    def apply_deaug_keypoints(self, keypoints, apply=False, **kwargs):
        if apply:
            keypoints = F.keypoints_vflip(keypoints)
        return keypoints
