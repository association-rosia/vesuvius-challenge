import os
import sys
sys.path.insert(0, os.pardir)

import torch
import cv2
from src.constant import TRAIN_FRAGMENTS_PATH, TEST_FRAGMENTS_PATH, TILE_SIZE
import numpy as np


def reconstruct_outputs(tiles, bboxes, fragments, fragments_shape):
    # Implementation of the reconstruction logic
    # Combine sub-masks to reconstruct the original images separately
    # Handle overlap by taking the mean of overlapping pixels

    reconstructed_outputs = {
        fragment: torch.zeros(mask_size).to(device=tiles.device)
        for fragment, mask_size in fragments_shape.items()
    }

    count_map = {
        fragment: torch.zeros(mask_size).to(device=tiles.device)
        for fragment, mask_size in fragments_shape.items()
    }

    for i in range(tiles.shape[0]):
        x0, y0, x1, y1 = bboxes[i]
        reconstructed_outputs[fragments[i]][y0:y1, x0:x1] += tiles[i, :, :]
        count_map[fragments[i]][y0:y1, x0:x1] += 1

    # Divide by the count map to obtain the mean value
    for key in fragments_shape.keys():
        reconstructed_image = reconstructed_outputs[key] / count_map[key]
        reconstructed_image = torch.nan_to_num(reconstructed_image, nan=0)

        shape = reconstructed_image.shape
        mask_path = os.path.join(TRAIN_FRAGMENTS_PATH, key, 'inklabels.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        padding = get_padding(mask.shape, TILE_SIZE)
        x0, y0, x1, y1 = padding[1][0], padding[0][0], shape[1] - padding[1][1], shape[0] - padding[0][1]
        reconstructed_image = reconstructed_image[y0:y1, x0:x1]
        reconstructed_outputs[key] = reconstructed_image

    return reconstructed_outputs


def get_fragment_shape(fragment_dir, fragment, tile_size):
    mask_path = os.path.join(fragment_dir, 'inklabels.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    padding = get_padding(mask.shape, tile_size)
    mask_pad = np.pad(mask, padding)

    return mask_pad.shape


# def get_fragments_shape(fragments, tile_size, test=False):
#     set_path = TRAIN_FRAGMENTS_PATH if not test else TEST_FRAGMENTS_PATH

#     return {fragment: get_fragment_shape(os.path.join(set_path, fragment), tile_size) for fragment in fragments}


def get_padding(mask_shape, tile_size, overlap=0.5):
    pad_left = int(overlap * tile_size)
    pad_up = int(overlap * tile_size)
    pad_right = int(overlap * tile_size + tile_size - mask_shape[1] % tile_size)
    pad_down = int(overlap * tile_size + tile_size - mask_shape[0] % tile_size)
    padding = [(pad_up, pad_down), (pad_left, pad_right)]

    return padding


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return torch.device(device=device)
