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
                      Z_START, Z_DIM, TILE_SIZE,
                      TRAIN_FRAGMENTS)

DEVICE = get_device()


def tile_fragment(set_path, fragment):
    fragment_path = os.path.join(set_path, fragment)
    image_shape = get_image_shape(set_path, fragment)
    image = np.zeros(shape=(Z_DIM, image_shape[0], image_shape[1]), dtype=np.uint8)
    image_path = sorted(glob.glob(os.path.join(fragment_path, 'surface_volume/*.tif')))[Z_START:Z_START + Z_DIM]

    for i, slice_path in enumerate(image_path):
        image[i, ...] = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)

    image_tiler = Tiler(
        data_shape=image.shape,
        tile_shape=(Z_DIM, TILE_SIZE, TILE_SIZE),
        overlap=0.5,
        channel_dimension=0,
    )

    new_shape, padding = image_tiler.calculate_padding()
    image_tiler.recalculate(data_shape=new_shape)
    image_pad = np.pad(image, padding)

    mask_path = os.path.join(fragment_path, 'inklabels.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask_tiler = Tiler(
        data_shape=mask.shape, tile_shape=(TILE_SIZE, TILE_SIZE), overlap=0.5
    )

    new_shape, padding = mask_tiler.calculate_padding()
    mask_tiler.recalculate(data_shape=new_shape)
    mask_pad = np.pad(mask, padding)

    tiles_zip = zip(image_tiler(image_pad), mask_tiler(mask_pad))

    fragment_list = []
    images = torch.ByteTensor().to(DEVICE)
    masks = torch.ByteTensor().to(DEVICE)
    bboxes = torch.IntTensor()

    for image_tile, mask_tile in tiles_zip:
        if mask_tile[1].max() > 0:
            print(f'Concat tile number {image_tile[0]} to main tensor from fragment {fragment}...')
            fragment_list.append(fragment)
            image = torch.unsqueeze(torch.from_numpy(image_tile[1]), dim=0).to(DEVICE)
            images = torch.cat((images, image), dim=0)
            mask = torch.unsqueeze(torch.from_numpy(mask_tile[1]), dim=0).to(DEVICE)
            masks = torch.cat((masks, mask), dim=0)

            bbox = image_tiler.get_tile_bbox(image_tile[0])
            bbox_tensor = torch.IntTensor([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])
            bboxes = torch.cat((bboxes, torch.unsqueeze(bbox_tensor, dim=0)), dim=0)

    return fragment_list, images, masks, bboxes


class CustomDataset(Dataset):
    def __init__(self, fragments, test, augmentation, multi_context):
        self.fragments = fragments
        self.test = test
        self.augmentation = augmentation
        self.multi_context = multi_context

        self.set_path = TRAIN_FRAGMENTS_PATH if not test else TEST_FRAGMENTS_PATH

        self.fragment_list = []
        self.images = torch.ByteTensor().to(DEVICE)
        self.masks = torch.ByteTensor().to(DEVICE)
        self.bboxes = torch.IntTensor()

        for fragment in fragments:
            fragment_list, images, masks, bboxes = tile_fragment(self.set_path, fragment)
            self.fragment_list += fragment_list
            self.images = torch.cat((self.images, images), dim=0)
            self.masks = torch.cat((self.masks, masks), dim=0)
            self.bboxes = torch.cat((self.bboxes, bboxes), dim=0)

        self.transforms = T.RandomApply(nn.ModuleList([T.RandomRotation(180),
                                                       T.RandomPerspective(),
                                                       T.ElasticTransform(alpha=500.0, sigma=10.0),
                                                       T.RandomHorizontalFlip(),
                                                       T.RandomVerticalFlip()]), p=0.5)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fragment = self.fragment_list[idx]
        image = self.images[idx] / 255.0
        mask = torch.unsqueeze(self.masks[idx] / 255.0, dim=0)
        bbox = self.bboxes[idx]  # [x0, y0, x1, y1]

        if self.augmentation:
            seed = random.randint(0, 2**32)
            torch.manual_seed(seed)
            image = self.transforms(image)
            torch.manual_seed(seed)
            mask = torch.squeeze(self.transforms(mask))

        return fragment, image, mask, bbox


def get_image_shape(set_path, fragment):
    fragment_path = os.path.join(set_path, fragment)
    mask_path = os.path.join(fragment_path, 'inklabels.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    return mask.shape


if __name__ == '__main__':
    train_dataset = CustomDataset(TRAIN_FRAGMENTS, test=False, augmentation=True, multi_context=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16)

    for fragment, image, mask, bbox in train_dataloader:
        print(fragment)
        print(image.shape)
        print(mask.shape)
        print(bbox.shape)
        break
