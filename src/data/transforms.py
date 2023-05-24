from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch

from ttach.base import DualTransform


from typing import List

# TODO
# T.RandomPerspective(),
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
        self.angle = float(torch.empty(1).uniform_(float(-degree), float(degree)).item())
        
        angles = [self.angle] if self.identity_param == self.angle else [self.identity_param, self.angle]

        super().__init__("angle", angles)

    def apply_aug_image(self, image, angle=0, interpolation=F.InterpolationMode.NEAREST, expand=False, center=None, fill=0, **kwargs):
        return F.rotate(image, angle, interpolation, expand, center, fill)

    def apply_deaug_mask(self, mask, angle=0, **kwargs):
        return self.apply_aug_image(mask, -angle)

    def apply_deaug_label(self, label, angle=0, **kwargs):
        return label
