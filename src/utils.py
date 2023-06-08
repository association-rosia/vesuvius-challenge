import os
import sys
sys.path.insert(0, os.pardir)

import torch
import cv2
from src.constant import TRAIN_FRAGMENTS_PATH, TILE_SIZE
import numpy as np


def reconstruct_output(tiles, bboxes, fragment_id, fragment_shape):
    reconstructed_output = torch.zeros(fragment_shape).to(device=tiles.device)
    count_map = torch.zeros(fragment_shape).to(device=tiles.device)

    for i in range(tiles.shape[0]):
        x0, y0, x1, y1 = bboxes[i]
        reconstructed_output[y0:y1, x0:x1] += tiles[i, :, :]
        count_map[y0:y1, x0:x1] += 1

    reconstructed_output /= count_map
    reconstructed_output = torch.nan_to_num(reconstructed_output, nan=0)

    mask_path = os.path.join(TRAIN_FRAGMENTS_PATH, fragment_id, 'inklabels.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    padding = get_padding(mask.shape, TILE_SIZE)

    shape = reconstructed_output.shape
    x0, y0, x1, y1 = padding[1][0], padding[0][0], shape[1] - padding[1][1], shape[0] - padding[0][1]
    reconstructed_output = reconstructed_output[y0:y1, x0:x1]

    return reconstructed_output


def get_fragment_shape(fragment_dir, fragment_id, tile_size):
    mask_path = os.path.join(fragment_dir, fragment_id, 'inklabels.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    padding = get_padding(mask.shape, tile_size)
    mask_pad = np.pad(mask, padding)

    return mask_pad.shape


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
