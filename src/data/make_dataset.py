import os
from os.path import join
import sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import glob
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import numpy as np
from tiler import Tiler

from src.utils import get_device
from constant import (TRAIN_FRAGMENTS_PATH, TEST_FRAGMENTS_PATH,
                      TRAIN_SAVE_PATH, TEST_SAVE_PATH,
                      Z_START, Z_DIM, TILE_SIZE,
                      TRAIN_FRAGMENTS)

DEVICE = get_device()


def tile_fragment(set_path, fragment, save_path, count):
    fragment_path = os.path.join(set_path, fragment)

    image_path = sorted(glob.glob(os.path.join(fragment_path, 'surface_volume/*.tif')))[Z_START:Z_START + Z_DIM]
    stack_list = [cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE) / 255.0 for slice_path in image_path]
    image = np.stack(stack_list, axis=0)

    image_tiler = Tiler(data_shape=image.shape,
                        tile_shape=(Z_DIM, TILE_SIZE, TILE_SIZE),
                        overlap=0.5,
                        channel_dimension=0)

    new_shape, padding = image_tiler.calculate_padding()
    image_tiler.recalculate(data_shape=new_shape)
    image_pad = np.pad(image, padding)
    mask_path = os.path.join(fragment_path, 'inklabels.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
    mask_tiler = Tiler(data_shape=mask.shape,
                       tile_shape=(TILE_SIZE, TILE_SIZE),
                       overlap=0.5)

    new_shape, padding = mask_tiler.calculate_padding()
    mask_tiler.recalculate(data_shape=new_shape)
    mask_pad = np.pad(mask, padding)
    tiles_zip = zip(image_tiler(image_pad), mask_tiler(mask_pad))

    tiles = []
    for image_tile, mask_tile in tiles_zip:
        if mask_tile[1].max() > 0:
            os.makedirs(join(save_path, str(count)), exist_ok=True)
            torch.save(torch.from_numpy(image_tile[1].astype('float32')), join(save_path, str(count), f'image.pt'))
            torch.save(torch.from_numpy(mask_tile[1].astype('float32')), join(save_path, str(count), f'mask.pt'))
            tiles.append({'tile': str(count), 'fragment': fragment})
            print(count)
            count += 1

    return tiles, count


class CustomDataset(Dataset):
    def __init__(self, fragments, test, augmentation, multi_context):
        self.tiles = []
        self.set_path = TRAIN_FRAGMENTS_PATH if not test else TEST_FRAGMENTS_PATH
        self.save_path = TRAIN_SAVE_PATH if not test else TEST_SAVE_PATH

        count = 0
        for fragment in fragments:
            tiles, count = tile_fragment(self.set_path, fragment, self.save_path, count)
            self.tiles += tiles

        self.augmentation = augmentation
        self.transforms = T.RandomApply(nn.ModuleList([T.RandomRotation(180),
                                                       T.RandomPerspective(),
                                                       T.ElasticTransform(alpha=500.0, sigma=10.0),
                                                       T.RandomHorizontalFlip(),
                                                       T.RandomVerticalFlip()]), p=0.5)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]['tile']
        fragment = self.tiles[idx]['fragment']

        image = torch.unsqueeze(torch.load(join(self.save_path, tile, f'image.pt'), map_location=DEVICE), dim=0)
        mask = torch.unsqueeze(torch.load(join(self.save_path, tile, f'mask.pt'), map_location=DEVICE), dim=0)

        if self.augmentation:
            seed = random.randint(0, 2 ** 32)
            torch.manual_seed(seed)
            image = self.transforms(image)
            torch.manual_seed(seed)
            mask = self.transforms(mask)

        return fragment, image, mask


#
# def get_mask_sizes(fragments):
#     mask_sizes = {}
#     for fragment in fragments:
#         fragment_path = os.path.join(FRAGMENTS_PATH, fragment)
#         mask_path = os.path.join(fragment_path, 'inklabels.png')
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask_sizes[fragment] = mask.shape
#
#     return mask_sizes


if __name__ == '__main__':
    train_dataset = CustomDataset(TRAIN_FRAGMENTS, test=False, augmentation=True, multi_context=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16)

    for indexes, inputs, masks in train_dataloader:
        print(indexes, inputs.shape, masks.shape)
        break
