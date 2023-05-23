import os
import sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import cv2
import numpy as np
import glob
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tiler import Tiler

from src.utils import get_device
from constant import Z_DIM, TILE_SIZE, TRAIN_FRAGMENTS, TRAIN_FRAGMENTS_PATH, TEST_FRAGMENTS_PATH

import matplotlib.pyplot as plt


def make_mask(fragment_path):
    mask_path = os.path.join(fragment_path, 'inklabels.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    shape = (Z_DIM, mask.shape[0], mask.shape[1])

    tiler = Tiler(data_shape=mask.shape,
                  tile_shape=(TILE_SIZE, TILE_SIZE),
                  overlap=0.5)

    new_shape, padding = tiler.calculate_padding()
    tiler.recalculate(data_shape=new_shape)
    mask_pad = np.pad(mask, padding)

    return tiler, mask_pad, shape, padding


def make_image(fragment_path, shape, padding):
    image = np.zeros(shape=shape, dtype=np.uint8)
    slices_path = sorted(glob.glob(os.path.join(fragment_path, 'surface_volume/*.tif')))[:Z_DIM]
    for i, slice_path in enumerate(slices_path):
        image[i, ...] = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)

    padding.insert(0, (0, 0))
    image_pad = np.pad(image, padding)

    return image_pad


class VesuviusDataset(Dataset):
    def __init__(self, fragments, test, threshold, augmentation, device):
        self.fragments = fragments
        self.test = test
        self.threshold = threshold
        self.augmentation = augmentation
        self.device = device
        self.set_path = TRAIN_FRAGMENTS_PATH if not test else TEST_FRAGMENTS_PATH
        self.data, self.items = self.make_data()

        self.transforms = T.RandomApply(
            nn.ModuleList([
                T.RandomRotation(180),
                T.RandomPerspective(),
                T.ElasticTransform(alpha=500.0, sigma=10.0),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip()
            ]), p=0.5
        )

    def make_data(self):
        data = {}
        items = []

        for fragment in self.fragments:
            fragment_path = os.path.join(self.set_path, str(fragment))
            tiler, mask_pad, shape, padding = make_mask(fragment_path)
            image_pad = make_image(fragment_path, shape, padding)
            items += self.get_items(fragment, tiler, mask_pad)

            data[fragment] = {
                'mask': torch.from_numpy(mask_pad).to(self.device),
                'image': torch.from_numpy(image_pad).to(self.device)
            }

        return data, items

    def get_items(self, fragment, tiler, mask_pad):
        items = []
        tiles = tiler(mask_pad)

        for tile in tiles:
            if tile[1].sum() / (255 * TILE_SIZE ** 2) >= self.threshold:
                bbox = tiler.get_tile_bbox(tile[0])
                bbox = torch.IntTensor([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])
                items.append({'fragment': fragment, 'bbox': bbox})

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fragment, bbox = self.items[idx]['fragment'], self.items[idx]['bbox']
        x0, y0, x1, y1 = bbox
        mask = torch.unsqueeze(self.data[fragment]['mask'][x0:x1, y0:y1] / 255.0, dim=0)
        image = self.data[fragment]['image'][:, x0:x1, y0:y1] / 255.0

        if self.augmentation:
            seed = random.randint(0, 2 ** 32)
            torch.manual_seed(seed)
            image = self.transforms(image)
            torch.manual_seed(seed)
            mask = torch.squeeze(self.transforms(mask))

        return fragment, bbox, mask, image


if __name__ == '__main__':
    DEVICE = get_device()
    train_dataset = VesuviusDataset(fragments=TRAIN_FRAGMENTS,
                                    test=False,
                                    threshold=0.01,
                                    augmentation=True,
                                    device=DEVICE)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16)
    for fragment, bbox, mask, image in train_dataloader:
        print(fragment)
        print(bbox.shape)
        print(mask.shape)
        print(image.shape)
        break
