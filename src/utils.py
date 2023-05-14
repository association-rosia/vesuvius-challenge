import os, sys

sys.path.insert(0, os.pardir)

from typing import List, Dict

import torch
import numpy as np

from constant import TILE_SIZE


def reconstruct_images(
    sub_masks: torch.Tensor,
    center_coords: torch.Tensor,
    fragment_ids: List,
    mask_sizes: Dict,
):
    # Implementation of the reconstruction logic
    # Combine sub-masks to reconstruct the original images separately
    # Handle overlap by taking the mean of overlapping pixels

    reconstructed_images = {
        fragment_id: torch.zeros(mask_size).to(device=sub_masks.device)
        for fragment_id, mask_size in mask_sizes.items()
    }
    count_map = {
        fragment_id: torch.zeros(mask_size).to(device=sub_masks.device)
        for fragment_id, mask_size in mask_sizes.items()
    }

    for i in range(sub_masks.shape[0]):
        x_center, y_center = center_coords[i, 0], center_coords[i, 1]

        # Calculate the start and end coordinates of the sub-mask
        x_start = x_center - TILE_SIZE // 2
        x_end = x_start + TILE_SIZE
        y_start = y_center - TILE_SIZE // 2
        y_end = y_start + TILE_SIZE

        # Handle overlap by taking the mean of overlapping pixels
        reconstructed_images[fragment_ids[i]][
            y_start:y_end, x_start:x_end
        ] += sub_masks[i, :, :]
        count_map[fragment_ids[i]][y_start:y_end, x_start:x_end] += 1

    # Divide by the count map to obtain the mean value
    for key in mask_sizes.keys():
        reconstructed_images[key] /= count_map[key]

    return reconstructed_images


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return torch.device(device=device)
