import os
import sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import glob
import cv2
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import numpy as np
from tiler import Tiler

from src.utils import get_device
from constant import FRAGMENTS_PATH, TRAIN_FRAGMENTS, Z_START, Z_DIM, TILE_SIZE

DEVICE = get_device()


def tile_fragment(fragment):
    fragment_path = os.path.join(FRAGMENTS_PATH, fragment)
    image_path = sorted(glob.glob(os.path.join(fragment_path, 'surface_volume/*.tif')))[Z_START:Z_START + Z_DIM]
    image = [cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE) / 255.0 for slice_path in image_path]
    image = np.stack(image, axis=0)

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
    tile_bbox_list = []
    tiles_zip = zip(image_tiler(image_pad), mask_tiler(mask_pad))

    for image_tile, mask_tile in tiles_zip:
        if mask_tile[1].max() > 0:
            fragment_list.append(torch.from_numpy(np.asarray([np.float32(fragment)])))
            image_list.append(torch.from_numpy(image_tile[1].astype('float32')))
            mask_list.append(torch.from_numpy(mask_tile[1].astype('float32')))
            tile_bbox_list.append(
                torch.from_numpy(np.array(image_tiler.get_tile_bbox(image_tile[0])).astype('float32'))
            )

    fragment = torch.stack(fragment_list, dim=0)
    image = torch.stack(image_list, dim=0)
    mask = torch.stack(mask_list, dim=0)
    tile_bbox = torch.stack(tile_bbox_list, dim=0)

    return fragment, image, mask, tile_bbox


class CustomDataset(Dataset):
    def __init__(self, fragments, augmentation):
        self.image = torch.Tensor()
        self.mask = torch.Tensor()
        self.tile_bbox = torch.Tensor()
        self.fragment = torch.Tensor()

        for fragment in fragments:
            fragment, image, mask, tile_bbox = tile_fragment(fragment)
            self.image = torch.cat((self.image, image), dim=0)
            self.mask = torch.cat((self.mask, mask), dim=0)
            self.tile_bbox = torch.cat((self.tile_bbox, tile_bbox), dim=0)
            self.fragment = torch.cat((self.fragment, fragment), dim=0)

        self.augmentation = augmentation
        self.transforms = torch.nn.Sequential(T.RandomRotation(180),
                                              T.RandomPerspective(),
                                              T.ElasticTransform(alpha=500.0, sigma=10.0),
                                              T.RandomHorizontalFlip(),
                                              T.RandomVerticalFlip())

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = torch.unsqueeze(self.image[idx], dim=0).to(DEVICE)
        mask = torch.unsqueeze(torch.unsqueeze(self.mask[idx], dim=0), dim=0).to(DEVICE)

        bbox = self.tile_bbox[idx]
        center = [(bbox[0, 0] + bbox[1, 0]) / 2, (bbox[0, 1] + bbox[1, 1]) / 2]
        center = torch.FloatTensor(center).to(DEVICE)

        fragment = self.fragment[idx].to(DEVICE)  # fragment id for reconstruction -> 1, 2 or 3

        if self.augmentation:
            seed = random.randint(0, 2 ** 32)
            torch.manual_seed(seed)
            image = self.transforms(image)
            torch.manual_seed(seed)
            mask = self.transforms(mask)

        return fragment, image, mask, center


if __name__ == '__main__':
    train_dataset = CustomDataset(TRAIN_FRAGMENTS, augmentation=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16)

    for indexes, inputs, masks, coords in train_dataloader:
        print(indexes, inputs.shape, masks.shape, coords)
        break
