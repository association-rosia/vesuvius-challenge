import os
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

import gc

from src.utils import get_device
from constant import (TRAIN_FRAGMENTS_PATH, TEST_FRAGMENTS_PATH,
                      Z_START, Z_DIM, TILE_SIZE,
                      TRAIN_FRAGMENTS)

DEVICE = get_device()


def tile_fragment(set_path, fragment):
    fragment_path = os.path.join(set_path, fragment)
    image_path = sorted(glob.glob(os.path.join(fragment_path, 'surface_volume/*.tif')))[Z_START:Z_START + Z_DIM]
    stack_list = [cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE) / 255.0 for slice_path in image_path]
    image = np.stack(stack_list, axis=0)

    del stack_list
    gc.collect()

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

    fragment_list = []
    image_list = []
    mask_list = []
    tiles_zip = zip(image_tiler(image_pad), mask_tiler(mask_pad))

    for image_tile, mask_tile in tiles_zip:
        if mask_tile[1].max() > 0:
            fragment_list.append(fragment)
            image_list.append(torch.from_numpy(image_tile[1].astype('float32')))
            mask_list.append(torch.from_numpy(mask_tile[1].astype('float32')))

    fragment = fragment_list
    image = torch.stack(image_list, dim=0)
    mask = torch.stack(mask_list, dim=0)

    del fragment_list, image_list, mask_list
    gc.collect()

    return fragment, image, mask


class CustomDataset(Dataset):
    def __init__(self, fragments, test, augmentation, multi_context):
        self.image = torch.Tensor()
        self.mask = torch.Tensor()
        self.fragment = []

        set_path = TRAIN_FRAGMENTS_PATH if not test else TEST_FRAGMENTS_PATH
        # save_path = TRAIN_SAVE_PATH if not test else TEST_SAVE_PATH
        for fragment in fragments:
            fragment, image, mask = tile_fragment(set_path, fragment)
            self.image = torch.cat((self.image, image), dim=0)
            self.mask = torch.cat((self.mask, mask), dim=0)
            self.fragment += fragment

        self.augmentation = augmentation
        self.transforms = T.RandomApply(nn.ModuleList([T.RandomRotation(180),
                                                       T.RandomPerspective(),
                                                       T.ElasticTransform(alpha=500.0, sigma=10.0),
                                                       T.RandomHorizontalFlip(),
                                                       T.RandomVerticalFlip()]), p=0.5)

    def __len__(self):
        return len(self.fragment)

    def __getitem__(self, idx):
        fragment = self.fragment[idx]
        image = torch.unsqueeze(self.image[idx], dim=0).to(DEVICE)
        mask = torch.unsqueeze(self.mask[idx], dim=0).to(DEVICE)  # don't remove it Baptiste

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
