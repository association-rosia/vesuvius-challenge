import os, sys

sys.path.insert(0, os.pardir)

from typing import List, Dict
import torch

import cv2

from constant import TRAIN_FRAGMENTS_PATH, TEST_FRAGMENTS_PATH


def reconstruct_images(sub_masks: torch.Tensor, bboxes: torch.Tensor, fragments: List, fragments_shape: Dict):
    # Implementation of the reconstruction logic
    # Combine sub-masks to reconstruct the original images separately
    # Handle overlap by taking the mean of overlapping pixels

    reconstructed_images = {
        fragment: torch.zeros(mask_size).to(device=sub_masks.device)
        for fragment, mask_size in fragments_shape.items()
    }

    count_map = {
        fragment: torch.zeros(mask_size).to(device=sub_masks.device)
        for fragment, mask_size in fragments_shape.items()
    }

    for i in range(sub_masks.shape[0]):
        x0, y0, x1, y1 = bboxes[i]
        reconstructed_images[fragments[i]][x0:x1, y0:y1] += sub_masks[i, :, :]
        count_map[fragments[i]][x0:x1, y0:y1] += 1

    # Divide by the count map to obtain the mean value
    for key in fragments_shape.keys():
        reconstructed_images[key] /= count_map[key]
        reconstructed_images[key] = torch.nan_to_num(reconstructed_images[key], nan=0)

    return reconstructed_images


def get_fragment_shape(fragment_dir, tile_size):
    mask_path = os.path.join(fragment_dir, 'inklabels.png')
    mask_shape = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).shape
    pad = tile_size // 2
    shape_pad = (mask_shape[0] + pad, mask_shape[1] + pad)
    return shape_pad


def get_fragments_shape(fragments, tile_size, test=False):
    set_path = TRAIN_FRAGMENTS_PATH if not test else TEST_FRAGMENTS_PATH
    return {fragment: get_fragment_shape(os.path.join(set_path, fragment), tile_size) for fragment in fragments}


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return torch.device(device=device)
